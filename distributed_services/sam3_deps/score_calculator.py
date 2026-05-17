"""
SAM3 得分计算器：根据不同类别计算得分
"""
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np

from .sam3_detector import SAM3Detector

def determine_position(
    locality: str,
    box1: Dict[str, float],
    box2: Dict[str, float],
    image_size: Optional[Tuple[int, int]] = None,
    iou_threshold: float = 0.1,
    distance_ratio: float = 0.15,
) -> float:
    """
    计算空间关系得分（参考 2D_spatial_eval_adapted.py）
    
    Args:
        locality: 空间关系词（如 "on the right of" 或 "below"）
        box1: 第一个对象的边界框 {"x_min": float, "y_min": float, "x_max": float, "y_max": float}
        box2: 第二个对象的边界框
        image_size: (width, height)，用于自适应距离阈值
        iou_threshold: IoU 阈值
        distance_ratio: 距离阈值占最长边的比例（默认 15%）
    
    Returns:
        得分 (0.0-1.0)
    """
    # 位置关系映射：新数据集格式 -> 评估代码期望格式
    locality_mapping = {
        "below": "on the bottom of",
        "above": "on the top of",
        "right of": "on the right of",
        "left of": "on the left of",
        "top of": "on the top of",
        "bottom of": "on the bottom of",
        "on right of": "on the right of",
        "on left of": "on the left of",
        "on top of": "on the top of",
        "on bottom of": "on the bottom of",
        # 保留原始格式（如果存在）
        "on the bottom of": "on the bottom of",
        "on the top of": "on the top of",
        "on the right of": "on the right of",
        "on the left of": "on the left of",
        "next to": "next to",
        "near": "near",
        "on side of": "on side of",
    }
    
    # 规范化 locality
    locality = locality.lower().strip()
    locality = locality_mapping.get(locality, locality)
    
    # 计算中心点
    box1_center = ((box1['x_min'] + box1['x_max']) / 2, (box1['y_min'] + box1['y_max']) / 2)
    box2_center = ((box2['x_min'] + box2['x_max']) / 2, (box2['y_min'] + box2['y_max']) / 2)
    
    # 计算距离
    x_distance = box2_center[0] - box1_center[0]
    y_distance = box2_center[1] - box1_center[1]
    max_dim = max(image_size) if image_size else 1024  # 默认兜底 1024，防止过大过小
    distance_threshold = max_dim * distance_ratio
    
    # 计算 IoU
    x_overlap = max(0, min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min']))
    y_overlap = max(0, min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min']))
    intersection = x_overlap * y_overlap
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0
    
    # 根据 locality 计算得分
    score = 0.0
    
    if locality in ['next to', 'on side of', 'near']:
        dist = min(abs(x_distance), abs(y_distance))
        score = max(0.0, min(1.0, distance_threshold / max(dist, 1e-5)))
    elif locality == 'on the right of':
        if x_distance < 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score = 1.0
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score = iou_threshold / iou if iou > 0 else 0.0
        else:
            score = 0.0
    elif locality == 'on the left of':
        if x_distance > 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score = 1.0
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score = iou_threshold / iou if iou > 0 else 0.0
        else:
            score = 0.0
    elif locality == 'on the bottom of':
        if y_distance < 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score = 1.0
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score = iou_threshold / iou if iou > 0 else 0.0
        else:
            score = 0.0
    elif locality == 'on the top of':
        if y_distance > 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score = 1.0
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score = iou_threshold / iou if iou > 0 else 0.0
        else:
            score = 0.0
    
    return score


class SAM3ScoreCalculator:
    """SAM3 得分计算器"""
    
    def __init__(self, detector: Optional[SAM3Detector] = None, device: str = "cuda"):
        """
        初始化得分计算器
        
        Args:
            detector: SAM3 检测器实例，如果为 None 则自动创建
            device: 设备
        """
        if detector is None:
            self.detector = SAM3Detector(device=device)
        else:
            self.detector = detector
    
    def calculate_spatial_score(
        self,
        image: Image.Image,
        obj1: str,
        obj2: str,
        locality: str
    ) -> float:
        """
        计算空间关系得分
        
        Args:
            image: PIL Image 对象
            obj1: 第一个对象名称
            obj2: 第二个对象名称
            locality: 空间关系词
        
        Returns:
            得分 (0.0-1.0)
        """
        if not self.detector.is_available():
            return 0.0
        
        # SAM3 是 zero-shot 模型，直接使用原始对象名称
        # locality 的规范化在 determine_position 函数内部完成
        
        # 检测对象
        object_names = [obj1, obj2]
        detected_objects, confidences, bboxes = self.detector.detect_objects(
            image,
            object_names,
            return_bbox=True,
            per_object_prompt=True,
        )
        
        # 匹配对象（SAM3 是 zero-shot，直接使用原始名称匹配）
        obj1_pos = None
        obj2_pos = None
        obj1_lower = obj1.lower()
        obj2_lower = obj2.lower()
        
        # 选匹配名称的最高置信度框
        obj1_best = (-1.0, None)  # (score, idx)
        obj2_best = (-1.0, None)
        
        for i, detected_obj in enumerate(detected_objects):
            detected_obj_lower = detected_obj.lower() if isinstance(detected_obj, str) else str(detected_obj).lower()
            score = confidences[i]
            
            if obj1_lower == detected_obj_lower or obj1_lower in detected_obj_lower or detected_obj_lower in obj1_lower:
                if score > obj1_best[0]:
                    obj1_best = (score, i)
            if obj2_lower == detected_obj_lower or obj2_lower in detected_obj_lower or detected_obj_lower in obj2_lower:
                if score > obj2_best[0]:
                    obj2_best = (score, i)
        
        obj1_pos = obj1_best[1]
        obj2_pos = obj2_best[1]
        
        # 部分匹配：按已匹配的对象置信度给部分分（每个 0.25）
        if obj1_pos is None or obj2_pos is None:
            partial_score = 0.0
            if obj1_pos is not None:
                partial_score += 0.25 * confidences[obj1_pos]
            if obj2_pos is not None:
                partial_score += 0.25 * confidences[obj2_pos]
            return partial_score
        
        # 计算空间关系得分
        box1 = {
            "x_min": bboxes[obj1_pos][0],
            "y_min": bboxes[obj1_pos][1],
            "x_max": bboxes[obj1_pos][2],
            "y_max": bboxes[obj1_pos][3],
        }
        box2 = {
            "x_min": bboxes[obj2_pos][0],
            "y_min": bboxes[obj2_pos][1],
            "x_max": bboxes[obj2_pos][2],
            "y_max": bboxes[obj2_pos][3],
        }
        
        position_score = determine_position(
            locality,
            box1,
            box2,
            image_size=(image.width, image.height),
        )
        
        # 综合得分：对象检测得分 + 空间关系得分
        obj_score = 0.25 * confidences[obj1_pos] + 0.25 * confidences[obj2_pos]
        spatial_score = position_score / 2
        score = obj_score + spatial_score

        return score
    
    def calculate_numeracy_score(
        self,
        image: Image.Image,
        expected_objects: List[str],
        expected_counts: List[int]
    ) -> float:
        """
        计算数量得分（参考 numeracy_eval_adapted.py）
        
        Args:
            image: PIL Image 对象
            expected_objects: 期望的对象名称列表
            expected_counts: 期望的数量列表
        
        Returns:
            得分 (0.0-1.0)
        """
        if not self.detector.is_available():
            return 0.0
        
        if len(expected_objects) != len(expected_counts):
            return 0.0
        
        # SAM3 是 zero-shot 模型，直接使用原始对象名称
        # 检测对象
        detected_objects, confidences, _ = self.detector.detect_objects(
            image,
            expected_objects,
            return_bbox=False,
            per_object_prompt=True,
        )
        
        # 计算得分
        score = 0.0
        weight = 1.0 / len(expected_objects)
        
        for i, expected_obj in enumerate(expected_objects):
            # 尝试匹配
            detected_count = 0
            matched_confidences: List[float] = []
            expected_obj_lower = expected_obj.lower() if isinstance(expected_obj, str) else str(expected_obj).lower()
            
            for idx, detected_obj in enumerate(detected_objects):
                detected_obj_lower = detected_obj.lower() if isinstance(detected_obj, str) else str(detected_obj).lower()
                if expected_obj_lower == detected_obj_lower or expected_obj_lower in detected_obj_lower or detected_obj_lower in expected_obj_lower:
                    detected_count += 1
                    matched_confidences.append(confidences[idx])
            
            # 存在与数量精准各占 50% 权重，既不稀疏也更敏感
            expected_num = max(1, expected_counts[i])
            presence_score = np.mean(matched_confidences) if detected_count > 0 else 0.0
            count_score = 1 if (matched_confidences and detected_count == expected_num) else 0.0
          
            score += weight * (0.5 * presence_score + 0.5 * count_score)
        
        return score
    
    def calculate_object_score(
        self,
        image: Image.Image,
        expected_objects: List[str],
    ) -> float:
        """
        计算对象存在得分（参考 object_eval_adapted.py）
        
        Args:
            image: PIL Image 对象
            expected_objects: 期望的对象名称列表（可以是普通对象名称，也可以是带属性的描述如 "purple elephant"）
        
        Returns:
            得分 (0.0-1.0)，表示检测到的对象数量 / 期望的对象数量
        """
        if not self.detector.is_available():
            return 0.0
        
        # SAM3 是 zero-shot 模型，直接使用原始对象名称
        # 检测对象
        detected_objects, confidences, _ = self.detector.detect_objects(
            image,
            expected_objects,
            return_bbox=False,
            per_object_prompt=True,
        )
        
        # 计算匹配的对象数量
        detected_count = 0
        conf_accum = 0.0
        for expected_obj in expected_objects:
            expected_obj_lower = expected_obj.lower() if isinstance(expected_obj, str) else str(expected_obj).lower()
            for idx, detected_obj in enumerate(detected_objects):
                detected_obj_lower = detected_obj.lower() if isinstance(detected_obj, str) else str(detected_obj).lower()
                if expected_obj_lower == detected_obj_lower or expected_obj_lower in detected_obj_lower or detected_obj_lower in expected_obj_lower:
                    detected_count += 1
                    conf_accum += confidences[idx]
                    break  # 每个期望对象只匹配一次
        
        # 得分：存在比例 × 置信度（避免 >1，且置信度直接作为信号）
        if len(expected_objects) > 0:
            base_score = detected_count / len(expected_objects)
            avg_conf = conf_accum / detected_count if detected_count > 0 else 0.0
            score = base_score * avg_conf
        else:
            score = 0.0
        
        return score
    
    def calculate_complex_score(
        self,
        image: Image.Image,
        expected_objects: List[str],
        spatial_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        计算复杂任务得分
        
        Args:
            image: PIL Image 对象
            expected_objects: 期望的对象名称列表
            spatial_info: 可选的空间信息 {"obj1": str, "obj2": str, "locality": str}
        
        Returns:
            得分 (0.0-1.0)
        """
        if not self.detector.is_available():
            return 0.0
        
        # 先计算对象存在得分
        object_score = self.calculate_object_score(image, expected_objects)
        
        # 如果有空间信息，计算空间关系得分
        if spatial_info:
            obj1 = spatial_info.get("obj1", "")
            obj2 = spatial_info.get("obj2", "")
            locality = spatial_info.get("locality", "")
            
            if obj1 and obj2 and locality:
                spatial_score = self.calculate_spatial_score(image, obj1, obj2, locality)
                # 综合得分：对象得分 50% + 空间关系得分 50%
                return spatial_score
        
        # 没有空间信息，只返回对象得分
        return object_score
    
    def calculate_score(
        self,
        image: Image.Image,
        metadata: Dict[str, Any],
        category: str
    ) -> float:
        """
        根据类别计算得分（统一入口）
        
        Args:
            image: PIL Image 对象
            metadata: 元数据字典（包含 nouns, attr_nouns, spatial_info, numeracy_info 等）
            category: 任务类别
        
        Returns:
            得分 (0.0-1.0)
        """
        if category == "spatial":
            # spatial 类别只有两个物体（obj1 和 obj2），没有三个物体的情况
            spatial_info = metadata.get("spatial_info", {})
            obj1 = spatial_info.get("obj1", "")
            obj2 = spatial_info.get("obj2", "")
            locality = spatial_info.get("locality", "")
            if obj1 and obj2 and locality:
                return self.calculate_spatial_score(image, obj1, obj2, locality)
            return 0.0
        
        elif category == "numeracy":
            numeracy_info = metadata.get("numeracy_info", [])
            if not numeracy_info:
                return 0.0
            expected_objects = [item.get("obj_name", "") for item in numeracy_info]
            expected_counts = [item.get("num", 1) for item in numeracy_info]
            return self.calculate_numeracy_score(image, expected_objects, expected_counts)
        
        elif category in ["color", "shape", "texture"]:
            # 对于 color/shape/texture 任务，直接使用 attr_nouns（带属性的对象描述）
            # 例如 "purple elephant" 应该直接检测，而不是只检测 "elephant"
            # 因为任务就是要判断属性（颜色/形状/纹理）是否正确
            attr_nouns = metadata.get("attr_nouns", [])
            if attr_nouns:
                # 直接使用 attr_nouns 作为检测目标（SAM3 可以理解带属性的描述）
                return self.calculate_object_score(image, attr_nouns)
            
            # 如果没有 attr_nouns，回退到 nouns（但这种情况不应该发生）
            nouns = metadata.get("nouns", [])
            if nouns:
                return self.calculate_object_score(image, nouns)
            return 0.0
        
        elif category == "object":
            # 使用 nouns
            nouns = metadata.get("nouns", [])
            if nouns:
                return self.calculate_object_score(image, nouns)
            return 0.0
        
        elif category == "non":
            # 使用 nouns
            nouns = metadata.get("nouns", [])
            if nouns:
                return self.calculate_object_score(image, nouns)
            return 0.0
        
        elif category == "complex":
            # 使用 nouns，如果有 spatial_info 也计算空间关系
            nouns = metadata.get("nouns", [])
            spatial_info = metadata.get("spatial_info")
            return self.calculate_complex_score(image, nouns, spatial_info)
        
        else:
            # 未知类别，返回 0
            return 0.0

