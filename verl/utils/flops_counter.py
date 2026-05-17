

from typing import TYPE_CHECKING, List, Tuple

import torch


if TYPE_CHECKING:
    from transformers.models.llama.configuration_llama import LlamaConfig


def get_device_flops(unit: str = "T") -> float:
    def unit_convert(number: float, level: str):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number

        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1

        return number

    device_name = torch.cuda.get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type
    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12

    flops_unit = unit_convert(flops, unit)
    return flops_unit


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)
    """

    def __init__(self, config: "LlamaConfig"):
        _ESTIMATE_FUNC = {
            "llama": self._estimate_llama_flops,
            "qwen2": self._estimate_llama_flops,
            "qwen2_vl": self._estimate_llama_flops,
            "qwen2_5_vl": self._estimate_llama_flops,
            "qwen2_5_vl_text": self._estimate_llama_flops,  # OmniVerifier 基于 Qwen2.5-VL
            "llavaonevision1_5": self._estimate_llama_flops,
            "qwen3": self._estimate_llama_flops,
            "qwen3_vl": self._estimate_llama_flops,
            "qwen3_vl_moe": self._estimate_llama_flops,
            "llava": self._estimate_llama_flops,
            "llava_next": self._estimate_llama_flops,
            "llava_onevision": self._estimate_llama_flops,
        }

        if config.model_type not in _ESTIMATE_FUNC:
            print(f"Only support {_ESTIMATE_FUNC.keys()}, but got {config.model_type}. MFU will always be zero.")

        self.config = config
        self._estimate_flops = _ESTIMATE_FUNC.get(config.model_type, self._estimate_unknown_flops)

    def _estimate_unknown_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        return 0

    def _estimate_llama_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        # Qwen3VLConfig 使用 text_config 来存储文本模型的配置
        if hasattr(self.config, 'text_config') and self.config.text_config is not None:
            # Qwen3-VL 模型：从 text_config 获取配置
            text_config = self.config.text_config
            hidden_size = text_config.hidden_size
            vocab_size = text_config.vocab_size
            num_hidden_layers = text_config.num_hidden_layers
            num_key_value_heads = text_config.num_key_value_heads
            num_attention_heads = text_config.num_attention_heads
            intermediate_size = text_config.intermediate_size
        else:
            # 标准 LLaMA/Qwen2 模型：直接从 config 获取
            hidden_size = self.config.hidden_size
            vocab_size = self.config.vocab_size
            num_hidden_layers = self.config.num_hidden_layers
            num_key_value_heads = self.config.num_key_value_heads
            num_attention_heads = self.config.num_attention_heads
            intermediate_size = self.config.intermediate_size

        # 计算 head_dim 等变量（需要在 if-else 之后，确保所有路径都定义了基础变量）
        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen

        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        
        # 处理 delta_time 为0或None的情况
        if delta_time is None or delta_time <= 0:
            return 0.0
        
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens: List[int], delta_time: float) -> Tuple[float, float]:
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        # 处理 batch_seqlens 不是列表的情况
        if not isinstance(batch_seqlens, list):
            if isinstance(batch_seqlens, (int, float)):
                batch_seqlens = [int(batch_seqlens)]
            else:
                batch_seqlens = list(batch_seqlens) if hasattr(batch_seqlens, '__iter__') else [0]
        
        # 处理空列表或无效值
        if not batch_seqlens or delta_time is None or delta_time <= 0:
            return 0.0, get_device_flops()
        
        tokens_sum = sum(batch_seqlens)
        estimated_flops = self._estimate_flops(tokens_sum, batch_seqlens, delta_time)
        
        # 确保返回值不为 None
        if estimated_flops is None:
            estimated_flops = 0.0
        
        promised_flops = get_device_flops()
        if promised_flops is None or promised_flops == float("inf"):
            promised_flops = 1.0  # 避免除零错误
        
        return estimated_flops, promised_flops
