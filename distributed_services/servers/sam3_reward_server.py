#!/usr/bin/env python3
"""
SAM3 奖励服务
启动后常驻内存加载 SAM3（可指定本地权重路径），提供 HTTP 接口计算奖励。
"""

import argparse
import base64
import io
import os
import sys
import json
import logging
from typing import Optional

from flask import Flask, request, jsonify
from PIL import Image
import torch

# 将 distributed_services 根目录加入路径，按包方式导入 sam3_deps
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICES_ROOT = os.path.normpath(os.path.join(SERVER_DIR, ".."))
if SERVICES_ROOT not in sys.path:
    sys.path.insert(0, SERVICES_ROOT)

from sam3_deps.sam3_reward import initialize_sam3_reward, compute_sam3_reward  # type: ignore  # noqa: E402

app = Flask(__name__)

# 全局缓存
SAM3_DEVICE: Optional[torch.device] = None


def load_sam3(bpe_path: Optional[str], ckpt_path: Optional[str], metadata_jsonl: Optional[str], device: torch.device):
    """预加载 SAM3 模型与元数据"""
    global SAM3_DEVICE
    SAM3_DEVICE = device
    initialize_sam3_reward(
        metadata_jsonl_path=metadata_jsonl,
        device=str(device),
        bpe_path=bpe_path,
        checkpoint_path=ckpt_path,
    )
    print(f"[INFO] SAM3 preloaded on {device} (bpe={bpe_path}, ckpt={ckpt_path})")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "device": str(SAM3_DEVICE)})


def _decode_image(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@app.route("/compute_sam3_reward", methods=["POST"])
def compute_sam3_reward_endpoint():
    try:
        # 记录请求信息
        print(f"[REQUEST] POST /compute_sam3_reward | prompt: {request.remote_addr}", flush=True)
        
        data = request.get_json(force=True)
        image_b64 = data.get("image")
        prompt = data.get("prompt")
        category = data.get("category", "object")
        ground_truth = data.get("ground_truth")
        bpe_path = data.get("bpe_path")
        ckpt_path = data.get("checkpoint_path")
        
        # 记录请求参数（截断prompt避免日志过长）
        prompt_preview = prompt[:50] + "..." if prompt and len(prompt) > 50 else prompt
        print(f"[REQUEST] category={category}, prompt_preview={prompt_preview}, has_ground_truth={bool(ground_truth)}", flush=True)

        if not image_b64 or not prompt:
            print(f"[ERROR] Missing image or prompt", flush=True)
            return jsonify({"success": False, "score": 0.0, "error": "Missing image or prompt"}), 400

        image = _decode_image(image_b64)

        # 调用奖励计算（直接传 PIL 图像，元数据优先用 ground_truth）
        result = compute_sam3_reward(
            image_path=None,
            prompt=prompt,
            category=category,
            ground_truth=ground_truth,
            device=str(SAM3_DEVICE or "cuda"),
            bpe_path=bpe_path,
            checkpoint_path=ckpt_path,
            image=image,
        )
        # compute_sam3_reward 总是返回包含 error 字段的字典
        # 成功时 error=None，失败时 error=str，需要确保 JSON 序列化时 None 变成空字符串
        # 如果 success=False 但 error 为空/None，提供默认错误消息
        error_msg = result.get("error")
        if error_msg is None or error_msg == "":
            if not result.get("success", False):
                # 如果失败但没有错误消息，提供默认错误消息
                error_msg = "SAM3 reward computation failed (no error message provided)"
            else:
                error_msg = ""
        
        # 记录响应信息
        if result["success"]:
            print(f"[RESPONSE] Success | score={result['score']:.4f}", flush=True)
        else:
            print(f"[RESPONSE] Failed | error={str(error_msg)[:100]}", flush=True)
        
        return jsonify({"success": result["success"], "score": result["score"], "error": str(error_msg)})
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] SAM3 reward computation exception: {error_detail}", flush=True)
        return jsonify({"success": False, "score": 0.0, "error": str(e)}), 500


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 reward server")
    parser.add_argument("--bpe_path", type=str, default=os.environ.get("SAM3_BPE_PATH"))
    parser.add_argument("--ckpt_path", type=str, default=os.environ.get("SAM3_CKPT_PATH"))
    parser.add_argument("--metadata_jsonl", type=str, default=os.environ.get("SAM3_METADATA_JSONL"))
    parser.add_argument("--device", type=int, default=int(os.environ.get("SAM3_DEVICE", "0")))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5010)
    return parser.parse_args()


def main():
    args = parse_args()
    # 如果设置了 CUDA_VISIBLE_DEVICES，PyTorch 只能看到索引为 0 的 GPU
    # 因此需要使用 cuda:0 而不是原始的 GPU ID
    if torch.cuda.is_available():
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            # CUDA_VISIBLE_DEVICES 已设置，使用 cuda:0
            device = torch.device("cuda:0")
        else:
            # 未设置 CUDA_VISIBLE_DEVICES，使用指定的设备 ID
            device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    
    load_sam3(args.bpe_path, args.ckpt_path, args.metadata_jsonl, device)
    print(f"🚀 SAM3 reward server running on {args.host}:{args.port}, device={device}")
    
    # 配置Flask日志：记录HTTP请求
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)  # 记录INFO级别（HTTP请求）
    # 确保werkzeug日志输出到标准输出（而不是stderr）
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.disabled = False
    
    app.run(host=args.host, port=args.port, threaded=True, processes=1)


if __name__ == "__main__":
    main()

