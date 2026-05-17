#!/usr/bin/env python3
"""
元数据合并脚本：将 geneval_and_t2i_data_final.jsonl 中的元数据合并到训练数据的 ground_truth 中
"""
import json
import argparse
from pathlib import Path
from typing import Optional
from metadata_loader import load_metadata_loader


def merge_metadata_to_ground_truth_batch(
    train_data_path: str,
    metadata_jsonl_path: str,
    output_path: Optional[str] = None
):
    """
    批量合并元数据到训练数据的 ground_truth 中
    
    Args:
        train_data_path: 训练数据 JSON 文件路径
        metadata_jsonl_path: 元数据 JSONL 文件路径
        output_path: 输出文件路径（如果为 None，覆盖原文件）
    """
    # 加载元数据
    metadata_loader = load_metadata_loader(metadata_jsonl_path)
    if metadata_loader is None:
        print("[ERROR] 无法加载元数据")
        return
    
    # 加载训练数据
    print(f"[INFO] 加载训练数据: {train_data_path}")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"[INFO] 共 {len(train_data)} 条训练数据")
    
    # 合并元数据
    merged_count = 0
    for i, item in enumerate(train_data):
        ground_truth_str = item.get("ground_truth", "")
        if ground_truth_str:
            try:
                merged_gt = metadata_loader.merge_metadata_to_ground_truth(ground_truth_str)
                if merged_gt != ground_truth_str:
                    item["ground_truth"] = merged_gt
                    merged_count += 1
            except Exception as e:
                print(f"[WARN] 合并第 {i} 条数据失败: {e}")
                continue
        
        # 每处理 1000 条输出一次进度
        if (i + 1) % 1000 == 0:
            print(f"[INFO] 已处理 {i + 1}/{len(train_data)} 条数据，合并 {merged_count} 条")
    
    # 保存结果
    output_file = output_path or train_data_path
    print(f"[INFO] 保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] 完成！已合并 {merged_count} 条记录的元数据")


def main():
    parser = argparse.ArgumentParser(description="合并元数据到训练数据的 ground_truth 中")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="训练数据 JSON 文件路径"
    )
    parser.add_argument(
        "--metadata_jsonl",
        type=str,
        required=True,
        help="元数据 JSONL 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（如果为 None，覆盖原文件）"
    )
    
    args = parser.parse_args()
    
    merge_metadata_to_ground_truth_batch(
        train_data_path=args.train_data,
        metadata_jsonl_path=args.metadata_jsonl,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
