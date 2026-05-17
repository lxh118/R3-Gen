"""
SAM3 检测器封装：用于检测图像中的对象和空间关系
"""
import os
import sys
import torch
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("[WARN] SAM3 模块导入失败，SAM3 检测功能将不可用")

DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # 与 SAM3Processor 默认保持一致


class SAM3Detector:
    """SAM3 检测器封装类"""
    
    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        bpe_path: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """
        初始化 SAM3 检测器
        
        Args:
            device: 设备 ('cuda' 或 'cpu')
            confidence_threshold: SAM3Processor 内部置信度阈值
            bpe_path: 可选，自定义 BPE 词表路径
            checkpoint_path: 可选，自定义权重路径
        """
        self.device = device
        self.model = None
        self.processor = None
        
        if not SAM3_AVAILABLE:
            print("[WARN] SAM3 不可用，检测器将无法工作")
            return
        
        try:
            # 支持自定义权重路径，否则走 sam3 默认路径/环境变量
            if bpe_path or checkpoint_path:
                self.model = build_sam3_image_model(
                    bpe_path=bpe_path,
                    checkpoint_path=checkpoint_path,
                )
            else:
                self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)
            if device != "cpu" and torch.cuda.is_available():
                self.model = self.model.to(device)
            print(f"[INFO] SAM3 模型加载成功 (device: {device})")
        except Exception as e:
            print(f"[ERROR] SAM3 模型加载失败: {e}")
            self.model = None
            self.processor = None
    
    def is_available(self) -> bool:
        """检查 SAM3 是否可用"""
        return self.model is not None and self.processor is not None
    
    def detect_objects(
        self,
        image: Image.Image,
        object_names: List[str],
        return_bbox: bool = True,
        per_object_prompt: bool = True,
    ) -> Tuple[List[str], List[float], Optional[List[List[float]]]]:
        """
        使用 SAM3 检测图像中的指定对象
        
        Args:
            image: PIL Image 对象
            object_names: 要检测的对象名称列表（如 ["chair", "table"]）
            return_bbox: 是否返回边界框坐标
            per_object_prompt: 是否对每个对象单独发起一次文本提示（更稳）
        
        Returns:
            detected_objects: 检测到的对象名称列表
            confidences: 对应的置信度列表
            bboxes: 边界框列表（如果 return_bbox=True），格式为 [[x1, y1, x2, y2], ...]
        """
        if not self.is_available():
            return [], [], None if return_bbox else []
        
        try:
            # 转换图像格式（如果需要）
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")
            
            # 设置图像
            inference_state = self.processor.set_image(image)
            
            # 处理检测结果
            detected_objects = []
            confidences = []
            bboxes_list = []
            
            # 获取图像尺寸
            img_width, img_height = image.size
            
            prompts = object_names if per_object_prompt else [". ".join(object_names) + "."]
            
            # 单对象逐个提示，便于计数和属性/位置对齐
            for idx, prompt in enumerate(prompts):
                with torch.no_grad():
                    output = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=prompt
                    )
                
                boxes = output.get("boxes", [])
                scores = output.get("scores", [])

                # 防止 torch.Tensor 参与布尔判断导致报错，统一转列表并用长度判断
                if torch.is_tensor(boxes):
                    boxes = boxes.detach().cpu().tolist()
                if torch.is_tensor(scores):
                    scores = scores.detach().cpu().tolist()

                if len(boxes) == 0 or len(scores) == 0:
                    continue
                
                # 当一次只提示一个对象时，使用对应名称；否则保持旧逻辑
                matched_obj = prompt if per_object_prompt else (object_names[idx] if idx < len(object_names) else (object_names[0] if object_names else None))
                
                for box, score in zip(boxes, scores):
                    name_to_use = matched_obj
                    if name_to_use is None:
                        continue
                    
                    detected_objects.append(name_to_use)
                    confidences.append(float(score))
                    
                    if return_bbox and box is not None:
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                            bboxes_list.append(bbox)
                        elif hasattr(box, 'tolist'):
                            bbox = box.tolist()
                            if len(bbox) >= 4:
                                bboxes_list.append(bbox[:4])
                            else:
                                bboxes_list.append([0, 0, img_width, img_height])
                        else:
                            bboxes_list.append([0, 0, img_width, img_height])
            
            return detected_objects, confidences, bboxes_list if return_bbox else None
            
        except Exception as e:
            print(f"[ERROR] SAM3 检测失败: {e}")
            return [], [], None if return_bbox else []

