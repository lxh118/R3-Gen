#!/usr/bin/env python3
"""
BAGEL服务 - 基于BAGEL模型
支持多GPU负载均衡，提供以下功能：
- 文生图（Text-to-Image）
- 图片编辑（Image Editing）
"""

import argparse
import sys
import logging
import os

# 禁用输出缓冲，确保日志立即写入
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# 禁用tqdm进度条输出（避免日志文件过大）
os.environ["TQDM_DISABLE"] = "1"

import base64
import io
import json
from pathlib import Path

import torch
import torch.distributed as dist
from flask import Flask, request, jsonify
from PIL import Image

# 禁用tqdm（如果环境变量不起作用，直接导入并禁用）
try:
    from tqdm import tqdm
    # 创建一个禁用输出的tqdm类
    class SilentTqdm:
        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else kwargs.get('iterable', [])
            self.total = kwargs.get('total', len(self.iterable) if hasattr(self.iterable, '__len__') else None)
        
        def __iter__(self):
            return iter(self.iterable)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            return False
        
        def update(self, n=1):
            pass
        
        def close(self):
            pass
    
    # 替换tqdm模块的tqdm类
    import tqdm as tqdm_module
    tqdm_module.tqdm = SilentTqdm
except ImportError:
    pass

# 添加本地bagel_deps路径（代码隔离）
current_file = os.path.abspath(__file__)
# distributed_services/servers/bagel_edit_server.py
# -> ../bagel_deps
bagel_deps_dir = os.path.join(os.path.dirname(os.path.dirname(current_file)), "bagel_deps")
bagel_deps_dir = os.path.abspath(bagel_deps_dir)

if os.path.exists(bagel_deps_dir):
    if bagel_deps_dir not in sys.path:
        sys.path.insert(0, bagel_deps_dir)
else:
    raise RuntimeError(
        f"无法找到bagel_deps目录。请确保路径存在。\n"
        f"尝试的路径: {bagel_deps_dir}\n"
        f"当前文件位置: {os.path.abspath(__file__)}"
    )

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.qwen2 import Qwen2Tokenizer

app = Flask(__name__)

# 全局模型变量
MODEL = None
VAE_MODEL = None
TOKENIZER = None
NEW_TOKEN_IDS = None
DEVICE = None
MODEL_TYPE = "bagel"


def setup_distributed_if_needed():
    """如果在多GPU环境中，初始化分布式"""
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank
    
    # 如果没有设置LOCAL_RANK，检查CUDA_VISIBLE_DEVICES
    # 当设置了CUDA_VISIBLE_DEVICES时，PyTorch会将可见的GPU重新编号为0,1,2...
    # 所以我们应该使用cuda:0，但可以在日志中显示实际的物理GPU ID
    return 0


def load_bagel_model(model_path: str, device: torch.device, max_latent_size: int = 64):
    """加载BAGEL模型"""
    global MODEL, VAE_MODEL, TOKENIZER, NEW_TOKEN_IDS, MODEL_TYPE
    
    print(f"开始加载BAGEL模型...", flush=True)
    print(f"模型路径: {model_path}", flush=True)
    print(f"设备: {device}", flush=True)
    MODEL_TYPE = "bagel"
    
    # 配置加载
    print("加载LLM配置...", flush=True)
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    print("加载Vision配置...", flush=True)
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    
    print("加载VAE模型...", flush=True)
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    
    print("创建BAGEL配置...", flush=True)
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=max_latent_size,
    )
    
    # 模型初始化
    print("初始化模型结构...", flush=True)
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, bagel_config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    
    # Tokenizer
    print("加载Tokenizer...", flush=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    # 加载checkpoint
    checkpoint_path = os.path.join(model_path, "ema.safetensors")
    print(f"加载模型checkpoint: {checkpoint_path}", flush=True)
    print("这可能需要几分钟时间，请耐心等待...", flush=True)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_path,
        device_map={"": "cpu"},
        dtype=torch.float32,
    )
    print("将模型移动到GPU...", flush=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    vae_model = vae_model.to(device=device, dtype=torch.bfloat16).eval()
    
    MODEL = model
    VAE_MODEL = vae_model
    TOKENIZER = tokenizer
    NEW_TOKEN_IDS = new_token_ids
    
    print(f"[OK] BAGEL model loaded successfully on {device}", flush=True)
    print(f"模型加载完成，显存使用情况:", flush=True)
    import subprocess
    try:
        # 获取当前使用的物理GPU ID
        physical_gpu_id = None
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
            if cuda_visible.strip().isdigit():
                physical_gpu_id = int(cuda_visible.strip())
            elif ',' in cuda_visible:
                physical_gpu_id = int(cuda_visible.split(',')[0].strip())
        
        # 查询所有GPU的显存信息
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # 解析输出，找到当前GPU的信息
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpu_idx = int(parts[0])
                        mem_used = parts[1]
                        mem_total = parts[2]
                        
                        # 如果指定了物理GPU ID，只显示该GPU的信息
                        # 否则显示所有GPU的信息（但只显示一次）
                        if physical_gpu_id is not None:
                            if gpu_idx == physical_gpu_id:
                                print(f"GPU {gpu_idx} 显存: {mem_used}MB / {mem_total}MB", flush=True)
                                break
                        else:
                            # 如果没有指定物理GPU ID，显示所有GPU的信息（但只显示一次）
                            print(f"GPU {gpu_idx} 显存: {mem_used}MB / {mem_total}MB", flush=True)
    except Exception as e:
        # 如果查询失败，静默处理（不影响服务启动）
        pass


@torch.no_grad()
def edit_image_bagel(
    image: Image.Image,
    edit_prompt: str,
    resolution: int = 1024,
    num_timesteps: int = 50,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    cfg_interval: list = None,
    timestep_shift: float = 3.0,
    cfg_renorm_min: float = 0.0,
    cfg_renorm_type: str = "text_channel",
    resolution_scale: float = 1.0,  # 降低latent size以提升速度
    # Taylor加速相关参数
    enable_taylorseer: bool = True,
    fresh_threshold: int = 3,
    max_order: int = 6,
    first_enhance: int = 5,
) -> Image.Image:
    """使用BAGEL模型编辑图像（参考Bagel inferencer实现）
    
    关键理解：
    - VAE/VIT transform参数(1024,512,16)和(980,224,14)是固定的，用于处理输入图片
    - 真正影响速度的是latent size，由image_shapes决定
    - image_shapes传给prepare_vae_latent，计算: h,w = H//latent_downsample, W//latent_downsample
    - num_image_tokens = h * w，这就是noise的token数量，直接影响生成速度
    
    Args:
        resolution_scale: 降低latent size的缩放因子 (0, 1.0]
            - 1.0: 使用原始尺寸（最高质量，最慢速度）
            - 0.75: 降低到75%，约1.78倍速度提升，质量略有下降（推荐）
            - 0.625: 降低到62.5%，约2.56倍速度提升
            - 0.5: 降低到50%（最小安全值，对应512），约4倍速度提升，质量明显下降
            - 注意：代码会自动确保最终尺寸≥512，保持在训练分布范围内
    """
    from modeling.bagel.qwen2_navit import NaiveCache
    from data.transforms import ImageTransform
    from data.data_utils import pil_img2rgb
    import copy
    
    if cfg_interval is None:
        cfg_interval = [0.0, 1.0]
    
    device = DEVICE
    
    # 图片预处理（参考Bagel实现）
    image = pil_img2rgb(image)

    vae_transform = ImageTransform(1024, 512, 16)  # 长边≤1024, 短边≥512, stride=16
    vit_transform = ImageTransform(980, 224, 14)   # 长边≤980, 短边≥224, stride=14
    
    
    # 初始化context（参考Bagel init_gen_context）
    gen_context = {
        'kv_lens': [0],
        'ropes': [0],
        'past_key_values': NaiveCache(MODEL.config.llm_config.num_hidden_layers),
    }
    
    # 初始化CFG contexts
    cfg_text_context = copy.deepcopy(gen_context)
    cfg_img_context = copy.deepcopy(gen_context)
    
    # 步骤1: 添加图片（参考Bagel interleave_inference line 248-253）
    # 图片需要先resize（参考Bagel line 249）
    image = vae_transform.resize_transform(image)
    image_shapes = image.size[::-1]  # (H, W)
    
    # 优化：降低latent size以提升速度（通过降低image_shapes）
    # 
    # 重要理解：
    # 1. Transform逻辑：max_size=1024, min_size=512 意味着：
    #    - 如果图片长边>1024，会缩小到≤1024
    #    - 如果图片短边<512，会放大到≥512
    #    - 如果图片在512-1024范围内，保持原尺寸（或调整到符合stride）
    # 2. 训练时：模型见过各种尺寸的latent（在512-1024范围内变化），不是固定的1024
    # 3. 推理时降低分辨率：
    #    - 如果降低到512-1024范围内（如768），仍在训练分布内，应该是安全的
    #    - 如果降低到<512，可能超出训练分布，质量会明显下降
    #    - 建议：resolution_scale >= 0.5，确保最终尺寸≥512
    if resolution_scale < 1.0:
        H, W = image_shapes
        # 按比例降低，并确保能被latent_downsample整除
        new_H = int(H * resolution_scale) // MODEL.latent_downsample * MODEL.latent_downsample
        new_W = int(W * resolution_scale) // MODEL.latent_downsample * MODEL.latent_downsample
        
        # 安全检查：确保不会太小（至少是训练时的min_size，即512）
        # 这样可以保持在训练分布范围内
        min_safe_size = 512  # 对应训练时的min_image_size
        new_H = max(new_H, min_safe_size)
        new_W = max(new_W, min_safe_size)
        
        # 如果调整后尺寸变化，更新image_shapes和图片
        if new_H != H or new_W != W:
            image_shapes = (new_H, new_W)
            # 重新resize图片以匹配新的image_shapes（用于后续的VAE/VIT处理）
            image = image.resize((new_W, new_H), Image.BICUBIC)
            
            # 记录优化信息
            if not hasattr(edit_image_bagel, '_last_resolution_scale') or edit_image_bagel._last_resolution_scale != resolution_scale:
                h, w = new_H // MODEL.latent_downsample, new_W // MODEL.latent_downsample
                num_tokens = h * w
                orig_h, orig_w = H // MODEL.latent_downsample, W // MODEL.latent_downsample
                orig_tokens = orig_h * orig_w
                speedup = orig_tokens / num_tokens if num_tokens > 0 else 1.0
                print(f"[Latent优化] resolution_scale={resolution_scale}")
                print(f"  - 原始图片尺寸: {W}x{H} -> Latent tokens: {orig_tokens}")
                print(f"  - 优化后尺寸: {new_W}x{new_H} -> Latent tokens: {num_tokens}")
                print(f"  - 预期速度提升: 约{speedup:.2f}倍（基于latent token数量减少）")
                print(f"  - 注意：降低到{new_W}x{new_H}仍在训练分布范围内（512-1024），应该是安全的")
                edit_image_bagel._last_resolution_scale = resolution_scale
    
    # 更新图片到context（VAE和VIT）
    generation_input_vae, gen_context['kv_lens'], gen_context['ropes'] = MODEL.prepare_vae_images(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        images=[image],
        transforms=vae_transform,
        new_token_ids=NEW_TOKEN_IDS,
    )
    for key, value in generation_input_vae.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input_vae[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input_vae[key] = value.to(device=device)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_context['past_key_values'] = MODEL.forward_cache_update_vae(
            VAE_MODEL, gen_context['past_key_values'], **generation_input_vae
        )
    
    generation_input_vit, gen_context['kv_lens'], gen_context['ropes'] = MODEL.prepare_vit_images(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        images=[image],
        transforms=vit_transform,
        new_token_ids=NEW_TOKEN_IDS,
    )
    for key, value in generation_input_vit.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input_vit[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input_vit[key] = value.to(device=device)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_context['past_key_values'] = MODEL.forward_cache_update_vit(
            gen_context['past_key_values'], **generation_input_vit
        )
    
    # 在添加图片后，复制context用于CFG（参考Bagel line 253）
    cfg_text_context = copy.deepcopy(gen_context)
    cfg_img_context = copy.deepcopy(gen_context)
    
    # 步骤2: 添加编辑指令（参考Bagel line 244-246）
    generation_input, gen_context['kv_lens'], gen_context['ropes'] = MODEL.prepare_prompts(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        prompts=[edit_prompt],
        tokenizer=TOKENIZER,
        new_token_ids=NEW_TOKEN_IDS,
    )
    for key, value in generation_input.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input[key] = value.to(device=device)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_context['past_key_values'] = MODEL.forward_cache_update_text(
            gen_context['past_key_values'], **generation_input
        )
        # 同时更新cfg_img_context（参考Bagel line 246）
        cfg_img_context['past_key_values'] = MODEL.forward_cache_update_text(
            cfg_img_context['past_key_values'], **generation_input
        )
        # 重要：同步更新cfg_img_context的kv_lens和ropes，否则会导致形状不匹配
        cfg_img_context['kv_lens'] = gen_context['kv_lens']
        cfg_img_context['ropes'] = gen_context['ropes']
    
    # 步骤3: 准备生成输入和CFG输入（参考Bagel gen_image）
    generation_input = MODEL.prepare_vae_latent(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        image_sizes=[image_shapes],
        new_token_ids=NEW_TOKEN_IDS,
    )
    for key, value in generation_input.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input[key] = value.to(device=device)
    
    # CFG text context
    generation_input_cfg_text = MODEL.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_context['kv_lens'],
        curr_rope=cfg_text_context['ropes'],
        image_sizes=[image_shapes],
    )
    for key, value in generation_input_cfg_text.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input_cfg_text[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input_cfg_text[key] = value.to(device=device)
    
    # CFG img context
    generation_input_cfg_img = MODEL.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_context['kv_lens'],
        curr_rope=cfg_img_context['ropes'],
        image_sizes=[image_shapes],
    )
    for key, value in generation_input_cfg_img.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input_cfg_img[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input_cfg_img[key] = value.to(device=device)
    
    # 步骤4: 生成图像（直接使用 MODEL.generate_image，已支持完整的 Taylor 加速）
    # 临时禁用tqdm输出（避免日志文件过大）
    old_tqdm_disable = os.environ.get("TQDM_DISABLE", None)
    os.environ["TQDM_DISABLE"] = "1"
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        unpacked_latent = MODEL.generate_image(
            past_key_values=gen_context['past_key_values'],
            cfg_text_past_key_values=cfg_text_context['past_key_values'],
            cfg_img_past_key_values=cfg_img_context['past_key_values'],
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            enable_taylorseer=enable_taylorseer,
            fresh_threshold=fresh_threshold,
            max_order=max_order,
            first_enhance=first_enhance,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
        )
    
    # 步骤5: 解码latent（参考Bagel decode_image）
    H, W = image_shapes
    h, w = H // MODEL.latent_downsample, W // MODEL.latent_downsample
    
    latent = unpacked_latent[0].reshape(1, h, w, MODEL.latent_patch_size, MODEL.latent_patch_size, MODEL.latent_channel)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, MODEL.latent_channel, h * MODEL.latent_patch_size, w * MODEL.latent_patch_size)
    latent = latent.to(device=device, dtype=torch.bfloat16)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        decoded_image = VAE_MODEL.decode(latent)
    
    # 后处理（参考Bagel decode_image line 182-183）
    edited_image = (
        (decoded_image * 0.5 + 0.5).clamp(0, 1)[0]
        .permute(1, 2, 0)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    
    return Image.fromarray(edited_image)


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy", "model_type": MODEL_TYPE})


@app.route("/edit", methods=["POST"])
def edit_image_endpoint():
    """
    图像编辑API端点
    
    请求格式（JSON）:
    {
        "image": "base64编码的图像",
        "edit_prompt": "编辑指令",
        "resolution": 1024,  # 可选，已废弃（保留兼容性）
        "num_timesteps": 50,  # 可选
        "cfg_text_scale": 4.0,  # 可选（或使用cfg_scale，向后兼容）
        "cfg_img_scale": 2.0,  # 可选
        "cfg_interval": [0.0, 1.0],  # 可选
        "timestep_shift": 3.0,  # 可选
        "cfg_renorm_min": 0.0,  # 可选
        "cfg_renorm_type": "text_channel",  # 可选
        "resolution_scale": 0.75,  # 可选，降低latent size以提升速度 (0, 1.0]（默认0.75）
        "enable_taylorseer": true,  # 可选，是否启用Taylor加速（默认true）
        "fresh_threshold": 3,  # 可选，Taylor加速参数（默认3）
        "max_order": 6,  # 可选，Taylor加速参数（默认6）
        "first_enhance": 5,  # 可选，Taylor加速参数（默认5）
    }
    
    返回格式（JSON）:
    {
        "success": true,
        "image": "base64编码的编辑后图像",
        "error": null
    }
    
    关于resolution_scale（关键参数）:
    - 这个参数直接影响latent space的大小，也就是noise的token数量
    - 1.0: 使用原始尺寸，最高质量，最慢速度
    - 0.75: 降低到75%，约1.78倍速度提升，质量略有下降
    - 0.5: 降低到50%，约4倍速度提升，质量明显下降
    - 推荐: 0.75-0.875 之间，平衡速度和质量
    """
    try:
        data = request.get_json()
        
        # 解析输入
        image_b64 = data.get("image")
        edit_prompt = data.get("edit_prompt")
        
        if not image_b64 or not edit_prompt:
            return jsonify({
                "success": False,
                "image": None,
                "error": "Missing required fields: image or edit_prompt"
            }), 400
        
        # 解码图像
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 参数
        resolution = data.get("resolution", 1024)
        num_timesteps = data.get("num_timesteps", 50)
        cfg_text_scale = data.get("cfg_text_scale", data.get("cfg_scale", 4.0))  # 兼容旧参数名
        cfg_img_scale = data.get("cfg_img_scale", 2.0)
        # 确保 cfg_interval 是有效的列表（防止 None 或无效值）
        cfg_interval = data.get("cfg_interval")
        if not isinstance(cfg_interval, list) or len(cfg_interval) != 2:
            cfg_interval = [0.0, 1.0]
        timestep_shift = data.get("timestep_shift", 3.0)
        cfg_renorm_min = data.get("cfg_renorm_min", 0.0)
        cfg_renorm_type = data.get("cfg_renorm_type", "text_channel")
        resolution_scale = data.get("resolution_scale", 0.75)  # 降低latent size以提升速度
        # Taylor加速参数
        enable_taylorseer = data.get("enable_taylorseer", True)
        fresh_threshold = data.get("fresh_threshold", 3)
        max_order = data.get("max_order", 6)
        first_enhance = data.get("first_enhance", 5)
        
        # 执行编辑
        edited_image = edit_image_bagel(
            image=image,
            edit_prompt=edit_prompt,
            resolution=resolution,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            resolution_scale=resolution_scale,
            enable_taylorseer=enable_taylorseer,
            fresh_threshold=fresh_threshold,
            max_order=max_order,
            first_enhance=first_enhance,
        )
        
        # 编码返回
        buffer = io.BytesIO()
        edited_image.save(buffer, format="PNG")
        edited_image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return jsonify({
            "success": True,
            "image": edited_image_b64,
            "error": None
        })
        
    except Exception as e:
        # 打印详细错误信息到日志（便于调试）
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"[ERROR] /edit 端点处理失败: {error_msg}", flush=True)
        print(f"[ERROR] 错误堆栈:\n{error_traceback}", flush=True)
        
        return jsonify({
            "success": False,
            "image": None,
            "error": error_msg
        }), 500


def parse_args():
    parser = argparse.ArgumentParser(description="BAGEL服务（支持文生图和图片编辑）")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--port", type=int, default=5001, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务host")
    parser.add_argument("--max_latent_size", type=int, default=64, help="BAGEL latent size")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 立即输出启动信息（刷新缓冲）
    print("=" * 60, flush=True)
    print("BAGEL服务启动中...", flush=True)
    print(f"模型路径: {args.model_path}", flush=True)
    print("=" * 60, flush=True)
    
    try:
        # 设置分布式（如果需要）
        local_rank = setup_distributed_if_needed()
        device = torch.device(f"cuda:{local_rank}")
        
        # 获取实际的物理GPU ID（用于日志显示）
        # 当设置了CUDA_VISIBLE_DEVICES时，需要显示实际的物理GPU ID
        physical_gpu_id = None
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
            # CUDA_VISIBLE_DEVICES可能是单个数字或逗号分隔的列表
            if cuda_visible.strip().isdigit():
                physical_gpu_id = int(cuda_visible.strip())
            elif ',' in cuda_visible:
                # 如果是多个GPU，取第一个（因为每个服务实例只使用一个GPU）
                physical_gpu_id = int(cuda_visible.split(',')[0].strip())
        
        global DEVICE
        DEVICE = device
        
        # 显示设备信息
        if physical_gpu_id is not None:
            print(f"使用设备: {device} (物理GPU: {physical_gpu_id})", flush=True)
        else:
            print(f"使用设备: {device}", flush=True)
        
        # 加载模型
        print("开始加载模型...", flush=True)
        load_bagel_model(args.model_path, device, args.max_latent_size)
        
        # 启动服务
        print(f"[START] Starting BAGEL server on {args.host}:{args.port}", flush=True)
        print(f"   Model path: {args.model_path}", flush=True)
        if physical_gpu_id is not None:
            print(f"   Device: {device} (物理GPU: {physical_gpu_id})", flush=True)
        else:
            print(f"   Device: {device}", flush=True)
        
        # 配置Flask日志：只记录HTTP请求，不记录其他信息
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.INFO)  # 只记录INFO级别（HTTP请求）
        # 禁用Flask的调试日志
        logging.getLogger('werkzeug').disabled = False  # 保持HTTP请求日志
        
        # 从环境变量读取是否启用多线程（默认：True，保持向后兼容）
        bagel_threaded = os.environ.get("EDIT_SERVER_THREADED", "true").lower() == "true"
        print(f"   Threaded mode: {bagel_threaded}", flush=True)
        
        # 输出Taylor加速配置信息（在服务启动时显示，确保日志记录）
        print(f"   Taylor加速: [可配置] 默认启用，可通过请求参数 enable_taylorseer 控制", flush=True)
        print(f"   [TaylorSeer] 默认参数: fresh_threshold=3, max_order=6, first_enhance=5", flush=True)
        print(f"   [TaylorSeer] 可通过请求参数动态调整Taylor加速参数和resolution_scale", flush=True)
        
        app.run(host=args.host, port=args.port, threaded=bagel_threaded)
    except Exception as e:
        print(f"[ERROR] 服务启动失败: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
