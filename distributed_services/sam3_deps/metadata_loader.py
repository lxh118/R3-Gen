"""
元数据加载器：从 geneval_and_t2i_data_final.jsonl 中加载元数据
"""
import json
import os
from typing import Dict, Optional, List
from pathlib import Path


class MetadataLoader:
    """从 JSONL 文件加载元数据，支持根据 prompt 快速查找"""
    
    def __init__(self, jsonl_path: str):
        """
        初始化元数据加载器
        
        Args:
            jsonl_path: geneval_and_t2i_data_final.jsonl 的路径
        """
        self.jsonl_path = jsonl_path
        self.metadata_dict: Dict[str, Dict] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """加载所有元数据到内存（SAM3 不受 'a photo of' 影响，直接保存原始 prompt）"""
        if not os.path.exists(self.jsonl_path):
            print(f"[WARN] 元数据文件不存在: {self.jsonl_path}")
            return
        
        count = 0
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    metadata = json.loads(line)
                    prompt = metadata.get("prompt", "").strip()
                    if prompt:
                        # 直接使用 prompt 作为 key（SAM3 不受 "a photo of" 影响，不需要去除前缀）
                        self.metadata_dict[prompt] = metadata
                        count += 1
                except Exception as e:
                    print(f"[WARN] 解析元数据行失败: {e}")
                    continue
        
        print(f"[INFO] 已加载 {count} 条元数据记录")
    
    def get_metadata(self, prompt: str) -> Optional[Dict]:
        """
        根据 prompt 获取元数据
        
        Args:
            prompt: 图像生成的 prompt
        
        Returns:
            元数据字典，如果未找到返回 None
        """
        prompt = prompt.strip()
        
        # 直接匹配（SAM3 不受 "a photo of" 影响，直接匹配即可）
        if prompt in self.metadata_dict:
            return self.metadata_dict[prompt]
        
        # 如果直接匹配失败，尝试模糊匹配（包含关系）
        for key, metadata in self.metadata_dict.items():
            if key in prompt or prompt in key:
                return metadata
        
        return None
    
    def merge_metadata_to_ground_truth(self, ground_truth_str: str) -> str:
        """
        将元数据合并到 ground_truth JSON 字符串中
        
        Args:
            ground_truth_str: 原始的 ground_truth JSON 字符串
        
        Returns:
            合并后的 ground_truth JSON 字符串
        """
        try:
            gt_data = json.loads(ground_truth_str)
            prompt = gt_data.get("prompt", "")
            
            if not prompt:
                return ground_truth_str
            
            # 获取元数据
            metadata = self.get_metadata(prompt)
            if metadata:
                # 合并元数据到 ground_truth
                # 保留原有的字段，添加元数据字段
                for key, value in metadata.items():
                    # 跳过 task_type，因为与 category 重复
                    if key == "task_type":
                        # 如果 ground_truth 中没有 category，将 task_type 转换为 category
                        if "category" not in gt_data:
                            gt_data["category"] = value
                        continue
                    
                    # 不覆盖已有字段
                    if key not in gt_data:
                        gt_data[key] = value
            
            return json.dumps(gt_data, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] 合并元数据失败: {e}")
            return ground_truth_str


def load_metadata_loader(jsonl_path: Optional[str] = None) -> Optional[MetadataLoader]:
    """
    加载元数据加载器（单例模式）
    
    Args:
        jsonl_path: 元数据文件路径，如果为 None，使用默认路径
    
    Returns:
        MetadataLoader 实例，如果文件不存在返回 None
    """
    if jsonl_path is None:
        jsonl_path = os.environ.get("SAM3_METADATA_JSONL", "")
        if not jsonl_path:
            print("[WARN] 未设置 SAM3_METADATA_JSONL，跳过元数据加载")
            return None
    
    if not os.path.exists(jsonl_path):
        print(f"[WARN] 元数据文件不存在: {jsonl_path}")
        return None
    
    return MetadataLoader(jsonl_path)
