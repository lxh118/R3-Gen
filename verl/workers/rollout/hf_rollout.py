from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: np.ndarray | torch.Tensor, repeats: int) -> np.ndarray | torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    return np.repeat(value, repeats, axis=0)


def _stack_mm_inputs(mm_inputs_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not mm_inputs_list:
        return {}

    keys = set(mm_inputs_list[0].keys())
    for item in mm_inputs_list[1:]:
        if set(item.keys()) != keys:
            raise ValueError("Inconsistent multimodal inputs across batch.")

    stacked: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in mm_inputs_list]
        if all(isinstance(v, torch.Tensor) for v in values):
            shapes = {tuple(v.shape) for v in values}
            if len(shapes) != 1:
                raise ValueError(f"Inconsistent tensor shapes for multimodal key {key}: {sorted(shapes)}")
            stacked[key] = torch.stack(values, dim=0)
        else:
            stacked[key] = values

    return stacked


class HFRollout(BaseRollout):
    def __init__(
        self,
        module: torch.nn.Module,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        generation_config: Optional[GenerationConfig] = None,
    ):
        super().__init__()
        self.module = module
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.generation_config = generation_config
        self.pad_token_id = tokenizer.pad_token_id

    def _build_mm_inputs(self, batch_multi_modal_data: np.ndarray, meta_info: dict[str, Any]) -> dict[str, Any]:
        if self.processor is None:
            raise ValueError("processor is required for multimodal HF rollout.")

        min_pixels = meta_info["min_pixels"]
        max_pixels = meta_info["max_pixels"]
        video_fps = meta_info["video_fps"]

        module_for_config = self.module.module if hasattr(self.module, "module") else self.module
        model_type = getattr(getattr(module_for_config, "config", None), "model_type", "")
        processor_name = self.processor.__class__.__name__.lower()
        image_processor_name = getattr(getattr(self.processor, "image_processor", None), "__class__", type("", (), {})).__name__.lower()
        if (
            model_type == "llavaonevision1_5"
            or "llavaonevision" in processor_name
            or "llavaonevision" in image_processor_name
        ):
            batch_images: list[list[Any]] = []
            has_images = False
            has_videos = False
            for multi_modal_data in batch_multi_modal_data:
                images: list[Any] = []
                videos: list[Any] = []
                if multi_modal_data is not None:
                    if "images" in multi_modal_data:
                        has_images = True
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, min_pixels, max_pixels))
                    if "videos" in multi_modal_data:
                        has_videos = True
                        for video in multi_modal_data["videos"]:
                            videos.append(process_video(video, min_pixels, max_pixels, video_fps))
                if videos:
                    raise NotImplementedError("HF rollout for LLaVA-OneVision-1.5 currently supports image batches only.")
                batch_images.append(images)

            if has_images and not has_videos:
                return dict(self.processor.image_processor(batch_images, return_tensors="pt"))

        mm_inputs_list: list[dict[str, Any]] = []
        for multi_modal_data in batch_multi_modal_data:
            images, videos = [], []
            if multi_modal_data is None:
                mm_inputs_list.append({})
                continue

            if "images" in multi_modal_data:
                for image in multi_modal_data["images"]:
                    images.append(process_image(image, min_pixels, max_pixels))

            if "videos" in multi_modal_data:
                for video in multi_modal_data["videos"]:
                    videos.append(process_video(video, min_pixels, max_pixels, video_fps))

            if len(images) != 0:
                is_internvl = "InternVL" in self.processor.__class__.__name__
                ip_kwargs = {"images": images, "return_tensors": "pt"}
                if is_internvl:
                    ip_kwargs["crop_to_patches"] = True
                mm_inputs = dict(self.processor.image_processor(**ip_kwargs))
                if is_internvl:
                    num_patches = mm_inputs.pop("num_patches", None)
                    for k in ("image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
                        mm_inputs.pop(k, None)
                    total_patches = sum(num_patches) if num_patches else len(images)
                    mm_inputs["image_flags"] = torch.ones(total_patches, 1, dtype=torch.long)
            elif len(videos) != 0:
                mm_inputs = dict(
                    self.processor.image_processor(images=None, videos=videos, return_tensors="pt")
                )
                if "InternVL" in self.processor.__class__.__name__:
                    for k in ("num_patches", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
                        mm_inputs.pop(k, None)
            else:
                mm_inputs = {}

            mm_inputs_list.append(mm_inputs)

        return _stack_mm_inputs(mm_inputs_list)

    def _build_generation_kwargs(
        self, meta_info: dict[str, Any], eos_token_id: int, pad_token_id: int
    ) -> dict[str, Any]:
        temperature = meta_info.get("temperature", self.config.temperature)
        top_p = meta_info.get("top_p", self.config.top_p)
        top_k = meta_info.get("top_k", self.config.top_k)
        num_return_sequences = meta_info.get("n", self.config.n)

        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.response_length,
            "do_sample": temperature is not None and temperature > 0,
            "temperature": temperature,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id,
            "use_cache": True,
        }
        if top_p is not None and top_p < 1.0:
            kwargs["top_p"] = top_p
        if top_k is not None and top_k > 0:
            kwargs["top_k"] = top_k

        if not self.config.ignore_eos:
            kwargs["eos_token_id"] = eos_token_id

        return kwargs

    def _sample_next_token(self, logits: torch.Tensor, gen_kwargs: dict[str, Any]) -> torch.Tensor:
        if not gen_kwargs["do_sample"]:
            return torch.argmax(logits, dim=-1)

        temperature = gen_kwargs.get("temperature", 1.0)
        if temperature is not None and temperature > 0:
            logits = logits / temperature

        top_k = gen_kwargs.get("top_k", None)
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth_vals, float("-inf"))

        top_p = gen_kwargs.get("top_p", None)
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]

        non_tensor_batch = prompts.non_tensor_batch
        batch_multi_modal_data = non_tensor_batch.get("multi_modal_data", None)

        device = input_ids.device
        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if batch_multi_modal_data is not None:
            mm_inputs = self._build_mm_inputs(batch_multi_modal_data, prompts.meta_info)
            for key, value in mm_inputs.items():
                if isinstance(value, torch.Tensor):
                    mm_inputs[key] = value.to(device)

        pad_token_id = self.pad_token_id if self.pad_token_id is not None else eos_token_id
        gen_kwargs = self._build_generation_kwargs(prompts.meta_info, eos_token_id, pad_token_id)

        was_training = self.module.training
        self.module.eval()
        response_ids_list = []
        num_return_sequences = gen_kwargs["num_return_sequences"]

        for batch_idx in range(input_ids.size(0)):
            sample_input_ids = input_ids[batch_idx : batch_idx + 1]
            sample_attention_mask = attention_mask[batch_idx : batch_idx + 1]
            sample_position_ids = position_ids[batch_idx : batch_idx + 1]

            sample_mm_inputs = {}
            if batch_multi_modal_data is not None:
                sample_mm_inputs = self._build_mm_inputs(
                    np.array([batch_multi_modal_data[batch_idx]], dtype=object),
                    prompts.meta_info,
                )
                for key, value in sample_mm_inputs.items():
                    if isinstance(value, torch.Tensor):
                        sample_mm_inputs[key] = value.to(device)

            for _ in range(num_return_sequences):
                cur_input_ids = sample_input_ids.clone()
                cur_attention_mask = sample_attention_mask.clone()
                response_tokens: list[int] = []
                past_key_values = None

                for _step in range(self.config.response_length):
                    if past_key_values is None:
                        cache_position = torch.arange(cur_input_ids.shape[1], device=device, dtype=torch.long)
                    else:
                        cache_position = torch.tensor([cur_input_ids.shape[1] - 1], device=device, dtype=torch.long)

                    prepare_fn = (
                        self.module.module.prepare_inputs_for_generation
                        if hasattr(self.module, "module")
                        else self.module.prepare_inputs_for_generation
                    )
                    model_inputs = prepare_fn(
                        cur_input_ids,
                        past_key_values=past_key_values,
                        attention_mask=cur_attention_mask,
                        cache_position=cache_position,
                        use_cache=True,
                        **sample_mm_inputs,
                    )
                    model_inputs["return_dict"] = True

                    outputs = self.module(**model_inputs)
                    next_token = self._sample_next_token(outputs.logits[:, -1, :], gen_kwargs)
                    next_token_id = next_token.item()
                    response_tokens.append(next_token_id)

                    past_key_values = outputs.past_key_values
                    next_token = next_token.view(1, 1)
                    cur_input_ids = torch.cat((cur_input_ids, next_token), dim=-1)
                    cur_attention_mask = torch.cat(
                        (
                            cur_attention_mask,
                            torch.ones((1, 1), dtype=cur_attention_mask.dtype, device=device),
                        ),
                        dim=-1,
                    )

                    if (not self.config.ignore_eos) and next_token_id == eos_token_id:
                        break

                response_ids_list.append(response_tokens)
        if was_training:
            self.module.train()
        response_ids = VF.pad_2d_list_to_length(
            response_ids_list, pad_token_id, max_length=self.config.response_length
        ).to(device)

        batch_size = input_ids.size(0)
        if gen_kwargs["num_return_sequences"] > 1:
            batch_size *= gen_kwargs["num_return_sequences"]
            input_ids = _repeat_interleave(input_ids, gen_kwargs["num_return_sequences"])
            attention_mask = _repeat_interleave(attention_mask, gen_kwargs["num_return_sequences"])
            position_ids = _repeat_interleave(position_ids, gen_kwargs["num_return_sequences"])
            if batch_multi_modal_data is not None:
                batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, gen_kwargs["num_return_sequences"])

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            out_non_tensor = {"multi_modal_data": batch_multi_modal_data}
        else:
            out_non_tensor = {}

        return DataProto(batch=batch, non_tensor_batch=out_non_tensor, meta_info=prompts.meta_info)
