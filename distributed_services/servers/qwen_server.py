#!/usr/bin/env python3
"""
Qwen Image Edit服务 - 基于Qwen-Image-Edit-2511模型 + cache-dit加速
支持多GPU负载均衡，提供以下功能：
- 图片编辑（Image Editing）
- 集成cache-dit加速（DBCache + SCM Ultra + TaylorSeer）
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
import threading

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

from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
import cache_dit
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig
# 导入 pipeline 模块以便动态修改 VAE_IMAGE_SIZE
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus

app = Flask(__name__)

# 全局模型变量
MODEL = None
DEVICE = None
# 线程锁，保护模型调用和缓存重新配置（仅在多线程模式下使用）
MODEL_LOCK = threading.Lock()
USE_THREADING = False  # 是否使用多线程模式（在main函数中设置）


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


def load_qwen_model(
    model_path: str,
    device: torch.device,
    enable_cache: bool = True,
    cache_config: DBCacheConfig = None,
    calibrator_config: TaylorSeerCalibratorConfig = None,
):
    """加载Qwen Image Edit模型并启用cache-dit加速"""
    global MODEL, DEVICE
    
    print(f"开始加载Qwen Image Edit模型...", flush=True)
    print(f"模型路径: {model_path}", flush=True)
    print(f"设备: {device}", flush=True)
    
    # 加载模型
    print("加载QwenImageEditPlusPipeline...", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    
    # 启用cache-dit加速
    if enable_cache:
        print("启用cache-dit加速...", flush=True)
        
        if cache_config is None:
            cache_config = DBCacheConfig(
                Fn_compute_blocks=8,      
                Bn_compute_blocks=0,        
                residual_diff_threshold=0.08,  
                steps_computation_mask=[1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(50)],
                steps_computation_policy="dynamic",
            )
        
        if calibrator_config is None:
            calibrator_config = TaylorSeerCalibratorConfig(
                enable_calibrator=True,
                enable_encoder_calibrator=True,
                calibrator_type="taylorseer",
                calibrator_cache_type="residual",
                taylorseer_order=6,
            )
        
        cache_dit.enable_cache(
            pipe,
            cache_config=cache_config,
            calibrator_config=calibrator_config,
        )
        print("✓ cache-dit加速已启用（DBCache + TaylorSeer）", flush=True)
    
    MODEL = pipe
    DEVICE = device
    
    print(f"[OK] Qwen Image Edit model loaded successfully on {device}", flush=True)
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
def edit_image_qwen(
    image: Image.Image,
    edit_prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    true_cfg_scale: float = 4.0,
    negative_prompt: str = "blurry, ugly, low quality, distorted",
    steps_computation_policy: str = "dynamic",
    steps_computation_mask: list = None,
    # 使用SCM时的参数
    use_scm: bool = False,
    scm_policy: str = "ultra",  # slow, medium, fast, ultra
    # 分辨率缩放参数（控制内部处理分辨率，影响速度和质量）
    resolution_scale: float = 1.0,  # 分辨率缩放因子 (0, 1.0]，默认1.0（1024×1024）
) -> Image.Image:
    """使用Qwen模型编辑图像
    
    Args:
        resolution_scale: 分辨率缩放因子，影响速度和质量
            - 1.0: 最高质量，最慢速度（默认，约1024×1024）
            - 0.75: 中等质量，中等速度（约768×768，速度提升约1.8倍）
            - 0.5: 较低质量，最快速度（约512×512，速度提升约4倍）
            - 其他值: 灵活调整，如0.625、0.875等
            
            注意：
            - 此参数会同时影响 VAE_IMAGE_SIZE（目标面积）和输出图像的 height/width
            - Pipeline 会根据输入图像的宽高比自动调整实际尺寸，保持宽高比一致
            - 例如：resolution_scale=0.75，输入图像 800×600，则：
              * VAE_IMAGE_SIZE = 1024² × 0.75² = 589,824 像素
              * 实际尺寸 ≈ 768×576（保持 4:3 宽高比）
    """
    device = DEVICE
    
    # 动态修改 pipeline 的 VAE_IMAGE_SIZE（通过修改模块常量）
    # VAE_IMAGE_SIZE 是目标面积（像素数），pipeline 会根据宽高比自动计算实际尺寸
    # 使用 resolution_scale 缩放目标面积：target_area = 1024² × scale²
    base_vae_image_size = 1024 * 1024  # 基准面积
    target_vae_image_size = int(base_vae_image_size * resolution_scale * resolution_scale)
    
    # 使用线程锁保护（避免并发修改）
    if USE_THREADING:
        MODEL_LOCK.acquire()
    try:
        original_vae_image_size = pipeline_qwenimage_edit_plus.VAE_IMAGE_SIZE
        
        if target_vae_image_size != original_vae_image_size:
            pipeline_qwenimage_edit_plus.VAE_IMAGE_SIZE = target_vae_image_size
            if not hasattr(edit_image_qwen, '_resolution_scale_logged') or edit_image_qwen._resolution_scale_logged != resolution_scale:
                print(f"[INFO] 设置 resolution_scale = {resolution_scale}", flush=True)
                print(f"  - VAE_IMAGE_SIZE (目标面积) = {target_vae_image_size} 像素", flush=True)
                print(f"  - Pipeline 会根据输入图像宽高比自动计算实际尺寸", flush=True)
                if resolution_scale < 1.0:
                    speedup = 1.0 / (resolution_scale * resolution_scale)
                    print(f"  - 预期速度提升: 约 {speedup:.1f}倍（相比 resolution_scale=1.0）", flush=True)
                edit_image_qwen._resolution_scale_logged = resolution_scale
    finally:
        if USE_THREADING:
            MODEL_LOCK.release()
    
    # 缓存配置管理（使用线程锁保护，避免并发请求时状态混乱）
    if MODEL is not None:
        # 确定实际使用的mask
        # 如果用户传入了mask，优先使用用户传入的mask；否则根据步数动态生成
        if steps_computation_mask is not None:
            # 用户传入了mask，使用用户指定的mask
            actual_mask = steps_computation_mask
            # 确保mask长度与步数匹配
            if len(actual_mask) != num_inference_steps:
                print(f"[WARNING] 传入的mask长度({len(actual_mask)})与步数({num_inference_steps})不匹配，将根据步数动态生成", flush=True)
                actual_mask = [1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(num_inference_steps)]
        elif use_scm:
            # 使用SCM时，根据策略生成mask
            actual_mask = cache_dit.steps_mask(
                mask_policy=scm_policy,
                total_steps=num_inference_steps,
            )
        else:
            # 不使用SCM且没有传入mask时，根据步数动态生成（每3步计算一次）
            actual_mask = [1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(num_inference_steps)]
        
        # 检查是否需要重新配置缓存
        # 只有当配置真正变化时才重载（步数、mask、SCM策略等）
        need_reconfig = False
        
        # 记录上次使用的配置（使用函数属性）
        if not hasattr(edit_image_qwen, '_last_steps'):
            edit_image_qwen._last_steps = 50 
        if not hasattr(edit_image_qwen, '_last_use_scm'):
            edit_image_qwen._last_use_scm = False
        if not hasattr(edit_image_qwen, '_last_scm_policy'):
            edit_image_qwen._last_scm_policy = "ultra"
        if not hasattr(edit_image_qwen, '_last_mask'):
            edit_image_qwen._last_mask = None
        
        # 检查配置是否变化
        if use_scm:
            # 使用SCM时，检查策略或步数是否变化
            if (not edit_image_qwen._last_use_scm or 
                edit_image_qwen._last_scm_policy != scm_policy or
                edit_image_qwen._last_steps != num_inference_steps):
                need_reconfig = True
                edit_image_qwen._last_use_scm = True
                edit_image_qwen._last_scm_policy = scm_policy
                edit_image_qwen._last_steps = num_inference_steps
                edit_image_qwen._last_mask = actual_mask
        else:
            # 不使用SCM时，检查步数或mask是否变化
            mask_changed = (edit_image_qwen._last_mask != actual_mask)
            steps_changed = (edit_image_qwen._last_steps != num_inference_steps)
            
            if mask_changed or steps_changed:
                need_reconfig = True
                edit_image_qwen._last_steps = num_inference_steps
                edit_image_qwen._last_use_scm = False
                edit_image_qwen._last_mask = actual_mask
        
        if need_reconfig:
            # 使用线程锁保护缓存重新配置（仅在多线程模式下需要，避免并发请求时状态混乱）
            if USE_THREADING:
                MODEL_LOCK.acquire()
            try:
                # 禁用当前缓存
                cache_dit.disable_cache(MODEL)
                
                if use_scm:
                    # 使用SCM配置（dynamic策略，SCM mask，order=1）
                    cache_config = DBCacheConfig(
                        Fn_compute_blocks=8,
                        Bn_compute_blocks=0,
                        residual_diff_threshold=0.08,
                        steps_computation_mask=actual_mask,
                        steps_computation_policy="dynamic",  # SCM使用动态策略
                    )
                    calibrator_config = TaylorSeerCalibratorConfig(
                        enable_calibrator=True,
                        enable_encoder_calibrator=True,
                        calibrator_type="taylorseer",
                        calibrator_cache_type="residual",
                        taylorseer_order=6, 
                    )
                else:
                    # 使用实际确定的mask（可能是用户传入的，也可能是动态生成的）
                    cache_config = DBCacheConfig(
                        Fn_compute_blocks=8,
                        Bn_compute_blocks=0,
                        residual_diff_threshold=0.08,
                        max_warmup_steps=5,  # 与初始配置一致
                        steps_computation_mask=actual_mask,  # 使用实际确定的mask
                        steps_computation_policy=steps_computation_policy, 
                    )
                    calibrator_config = TaylorSeerCalibratorConfig(
                        enable_calibrator=True,
                        enable_encoder_calibrator=True,
                        calibrator_type="taylorseer",
                        calibrator_cache_type="residual",
                        taylorseer_order=6,  # 默认使用order=6（与测试配置一致）
                    )
                
                # 重新启用缓存
                cache_dit.enable_cache(MODEL, cache_config=cache_config, calibrator_config=calibrator_config)
            finally:
                if USE_THREADING:
                    MODEL_LOCK.release()
    
    # 模型调用参数
    # 根据 resolution_scale 计算对应的 height/width，确保与 VAE_IMAGE_SIZE 一致
    # Pipeline 内部会根据 VAE_IMAGE_SIZE 和输入图像宽高比计算 vae_images 尺寸
    # 我们需要确保输出的 height/width 与 VAE_IMAGE_SIZE 的目标面积一致
    image_aspect_ratio = image.width / image.height
    # 计算目标面积（基于 resolution_scale）
    import math
    base_area = 1024 * 1024  # 基准面积
    target_area = int(base_area * resolution_scale * resolution_scale)
    
    # 使用与 pipeline 相同的 calculate_dimensions 逻辑计算尺寸
    # 这样可以确保与 pipeline 内部计算的 vae_images 尺寸一致
    calculated_width = math.sqrt(target_area * image_aspect_ratio)
    calculated_height = calculated_width / image_aspect_ratio
    
    # 确保是 32 的倍数（pipeline 要求，vae_scale_factor * 2 = 16 * 2 = 32）
    multiple_of = 32
    calculated_width = int(round(calculated_width / multiple_of) * multiple_of)
    calculated_height = int(round(calculated_height / multiple_of) * multiple_of)
    
    model_kwargs = {
        "prompt": edit_prompt,
        "negative_prompt": negative_prompt,
        "image": image,  # 使用原始图像
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "generator": torch.manual_seed(42),
        "height": calculated_height,  # 传递计算好的 height，确保与 VAE_IMAGE_SIZE 一致
        "width": calculated_width,     # 传递计算好的 width，确保与 VAE_IMAGE_SIZE 一致
        "guidance_scale": guidance_scale
    }
    
    # 使用线程锁保护模型调用（仅在多线程模式下需要，避免并发请求时状态混乱）
    if USE_THREADING:
        MODEL_LOCK.acquire()
    try:
        output = MODEL(**model_kwargs).images[0]
    finally:
        if USE_THREADING:
            MODEL_LOCK.release()
    
    return output


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy", "model_type": "qwen_image_edit"})


@app.route("/edit", methods=["POST"])
def edit_image_endpoint():
    """
    图像编辑API端点
    
    请求格式（JSON）:
    {
        "image": "base64编码的图像",
        "edit_prompt": "编辑指令",
        "num_inference_steps": 50,  # 可选
        "guidance_scale": 4.0,  # 可选
        "true_cfg_scale": 4.0,  # 可选
        "negative_prompt": "blurry, ugly, low quality, distorted",  # 可选
        "use_scm": false,  # 可选，是否使用SCM（Steps Computation Masking）
        "scm_policy": "ultra",  # 可选，SCM策略：slow, medium, fast, ultra
        "resolution_scale": 1.0,  # 可选，分辨率缩放因子 (0, 1.0]，影响速度和质量（默认1.0）
        "steps_computation_mask": [1,0,0,1,0,0,...],  # 可选，自定义计算mask（长度需匹配num_inference_steps）
        "steps_computation_policy": "dynamic",  # 可选，缓存策略：static 或 dynamic（默认dynamic）
    }
    
    关于resolution_scale（关键参数）:
    - 这个参数控制分辨率缩放，直接影响计算速度和输出质量
    - 工作原理：
      * VAE_IMAGE_SIZE (目标面积) = 1024² × resolution_scale²
      * Pipeline 会根据输入图像的宽高比自动计算实际尺寸
      * 输出的 height/width 也会根据相同的比例计算，确保一致
    - 推荐值：
      * 1.0: 最高质量，最慢速度（默认，约1024×1024，保持输入图像宽高比）
      * 0.75: 中等质量，中等速度（约768×768，速度提升约1.8倍，推荐）
      * 0.5: 较低质量，最快速度（约512×512，速度提升约4倍）
      * 其他值: 灵活调整，如0.625、0.875等，根据需求选择
    - 优势：
      * 灵活可控，不局限于固定尺寸
      * 自动保持输入图像的宽高比
      * 同时控制 VAE_IMAGE_SIZE 和输出 height/width，确保一致
    - 示例：
      * resolution_scale=0.75，输入图像 800×600 (4:3)
        → VAE_IMAGE_SIZE = 589,824 像素
        → 实际尺寸 ≈ 768×576 (保持 4:3 宽高比)
        → 输出 height/width = 768×576 (一致)
    
    关于steps_computation_mask:
    - 如果传入此参数，将使用你指定的mask，不会根据步数动态生成
    - 如果未传入，将根据num_inference_steps自动生成（每3步计算一次）
    - 只有当mask或步数变化时才会重新配置缓存，相同配置不会重复重载
    
    返回格式（JSON）:
    {
        "success": true,
        "image": "base64编码的编辑后图像",
        "error": null
    }
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
        num_inference_steps = data.get("num_inference_steps", 50)
        guidance_scale = data.get("guidance_scale", 1.0)
        true_cfg_scale = data.get("true_cfg_scale", 4.0)
        negative_prompt = data.get("negative_prompt", "blurry, ugly, low quality, distorted")
        use_scm = data.get("use_scm", False)
        scm_policy = data.get("scm_policy", "ultra")
        resolution_scale = data.get("resolution_scale", 1.0)  # 分辨率缩放因子
        steps_computation_mask = data.get("steps_computation_mask", None)  # 自定义mask（可选）
        steps_computation_policy = data.get("steps_computation_policy", "dynamic")  # 缓存策略（可选）
        
        # 验证 resolution_scale 参数
        if resolution_scale <= 0 or resolution_scale > 1.0:
            print(f"[WARNING] resolution_scale={resolution_scale} 超出有效范围 (0, 1.0]，已调整为 1.0", flush=True)
            resolution_scale = 1.0
        
        # 执行编辑
        edited_image = edit_image_qwen(
            image=image,
            edit_prompt=edit_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            steps_computation_policy=steps_computation_policy,
            steps_computation_mask=steps_computation_mask,
            use_scm=use_scm,
            scm_policy=scm_policy,
            resolution_scale=resolution_scale,  # 分辨率缩放因子
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
    parser = argparse.ArgumentParser(description="Qwen Image Edit服务（支持cache-dit加速）")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--port", type=int, default=5002, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务host")
    parser.add_argument("--disable_cache", action="store_true", help="禁用cache-dit加速")
    parser.add_argument("--taylorseer_order", type=int, default=6, help="TaylorSeer阶数（推荐1或2）")
    parser.add_argument("--residual_diff_threshold", type=float, default=0.08, help="残差差异阈值")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 立即输出启动信息（刷新缓冲）
    print("=" * 60, flush=True)
    print("Qwen Image Edit服务启动中...", flush=True)
    print(f"模型路径: {args.model_path}", flush=True)
    print("=" * 60, flush=True)
    
    try:
        # 设置分布式（如果需要）
        local_rank = setup_distributed_if_needed()
        device = torch.device(f"cuda:{local_rank}")
        
        # 获取实际的物理GPU ID（用于日志显示）
        physical_gpu_id = None
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
            if cuda_visible.strip().isdigit():
                physical_gpu_id = int(cuda_visible.strip())
            elif ',' in cuda_visible:
                physical_gpu_id = int(cuda_visible.split(',')[0].strip())
        
        global DEVICE
        DEVICE = device
        
        # 显示设备信息
        if physical_gpu_id is not None:
            print(f"使用设备: {device} (物理GPU: {physical_gpu_id})", flush=True)
        else:
            print(f"使用设备: {device}", flush=True)
        
        # 准备cache配置
        cache_config = None
        calibrator_config = None
        if not args.disable_cache:
            cache_config = DBCacheConfig(
                Fn_compute_blocks=8,
                Bn_compute_blocks=0,
                residual_diff_threshold=args.residual_diff_threshold,
                steps_computation_mask=[1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(50)],
                steps_computation_policy="dynamic",  # 静态缓存策略
            )
            calibrator_config = TaylorSeerCalibratorConfig(
                enable_calibrator=True,
                enable_encoder_calibrator=True,
                calibrator_type="taylorseer",
                calibrator_cache_type="residual",
                taylorseer_order=args.taylorseer_order,
            )
        
        # 加载模型
        print("开始加载模型...", flush=True)
        load_qwen_model(
            args.model_path,
            device,
            enable_cache=not args.disable_cache,
            cache_config=cache_config,
            calibrator_config=calibrator_config,
        )
        
        # 启动服务
        print(f"[START] Starting Qwen Image Edit server on {args.host}:{args.port}", flush=True)
        print(f"   Model path: {args.model_path}", flush=True)
        if physical_gpu_id is not None:
            print(f"   Device: {device} (物理GPU: {physical_gpu_id})", flush=True)
        else:
            print(f"   Device: {device}", flush=True)
        print(f"   Cache-dit: {'启用' if not args.disable_cache else '禁用'}", flush=True)
        if not args.disable_cache:
            print(f"   TaylorSeer order: {args.taylorseer_order}", flush=True)
            print(f"   Residual diff threshold: {args.residual_diff_threshold}", flush=True)
        
        # 配置Flask日志：只记录HTTP请求，不记录其他信息
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.INFO)  # 只记录INFO级别（HTTP请求）
        logging.getLogger('werkzeug').disabled = False  # 保持HTTP请求日志
        
        # 从环境变量读取是否启用多线程（默认：False，单线程模式性能更好且无并发问题）
        qwen_threaded = os.environ.get("EDIT_SERVER_THREADED", "false").lower() == "true"
        global USE_THREADING
        USE_THREADING = qwen_threaded
        print(f"   Threaded mode: {qwen_threaded}", flush=True)
        if qwen_threaded:
            print(f"   ⚠️  多线程模式已启用，将使用线程锁保护模型调用", flush=True)
        else:
            print(f"   ✓ 单线程模式，无需线程锁，性能最优", flush=True)
        
        app.run(host=args.host, port=args.port, threaded=qwen_threaded)
    except Exception as e:
        print(f"[ERROR] 服务启动失败: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

