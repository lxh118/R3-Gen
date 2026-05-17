"""
SAM3 奖励计算模块：集成到奖励函数中
"""
import json
import os
from typing import Dict, Optional, Any
from PIL import Image

# 仅包内导入（服务器以包方式加载）
from .metadata_loader import MetadataLoader, load_metadata_loader
from .sam3_detector import SAM3Detector
from .score_calculator import SAM3ScoreCalculator

# 全局实例（延迟初始化）
_metadata_loader: Optional[MetadataLoader] = None
_score_calculator: Optional[SAM3ScoreCalculator] = None


def initialize_sam3_reward(
    metadata_jsonl_path: Optional[str] = None,
    device: str = "cuda",
    bpe_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
):
    """
    初始化 SAM3 奖励计算模块
    
    Args:
        metadata_jsonl_path: 元数据 JSONL 文件路径
        device: 设备
        bpe_path: 可选，自定义 BPE 路径
        checkpoint_path: 可选，自定义权重路径
    """
    global _metadata_loader, _score_calculator
    
    # 仅在提供路径时加载元数据（可选）
    if _metadata_loader is None and metadata_jsonl_path:
        _metadata_loader = load_metadata_loader(metadata_jsonl_path)
    
    # 初始化得分计算器
    if _score_calculator is None:
        detector = SAM3Detector(
            device=device,
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
        )
        _score_calculator = SAM3ScoreCalculator(detector=detector, device=device)
    
    return _metadata_loader, _score_calculator


def compute_sam3_reward(
    image_path: Optional[str],
    prompt: str,
    category: str,
    ground_truth: Optional[str] = None,
    metadata_jsonl_path: Optional[str] = None,
    device: str = "cuda",
    bpe_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """
    计算 SAM3 奖励分数
    
    Args:
        image_path: 图像路径
        prompt: 图像生成的 prompt
        category: 任务类别（spatial, numeracy, color, shape, texture, object, non, complex）
        ground_truth: ground_truth JSON 字符串（可选，如果提供会从中提取元数据）
        metadata_jsonl_path: 元数据 JSONL 文件路径（可选）
        device: 设备
        image: 直接传入的 PIL Image（可选，优先使用）
    
    Returns:
        {
            "score": float,  # 得分 (0.0-1.0)
            "success": bool,  # 是否成功
            "error": str,  # 错误信息（如果有）
        }
    """
    try:
        # 初始化模块
        metadata_loader, score_calculator = initialize_sam3_reward(
            metadata_jsonl_path,
            device,
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
        )
        if not score_calculator.detector.is_available():
            return {
                "score": 0.0,
                "success": False,
                "error": "SAM3 detector not available"
            }
        
        # 加载图像（优先使用传入的 PIL Image，否则用路径）
        if image is None:
            if not image_path or not os.path.exists(image_path):
                return {
                    "score": 0.0,
                    "success": False,
                    "error": f"Image not found: {image_path}"
                }
            image = Image.open(image_path).convert("RGB")
        
        # 获取元数据
        metadata = None
        
        # 优先从 ground_truth 中获取元数据（如果已合并）
        if ground_truth:
            try:
                gt_data = json.loads(ground_truth)
                # 检查是否已包含元数据字段
                if "nouns" in gt_data or "spatial_info" in gt_data or "numeracy_info" in gt_data:
                    metadata = gt_data
            except:
                pass
        
        # 如果 ground_truth 中没有元数据：仅当外部显式提供 metadata_loader 时再尝试；否则直接返回缺失
        if metadata is None:
            if metadata_loader is not None:
                metadata = metadata_loader.get_metadata(prompt)
            if metadata is None:
                return {
                    "score": 0.0,
                    "success": False,
                    "error": f"Metadata not found for prompt: {prompt}"
                }
        
        # 计算得分
        score = score_calculator.calculate_score(image, metadata, category)
        
        return {
            "score": score,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "success": False,
            "error": str(e)
        }