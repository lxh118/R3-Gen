# SAM3 奖励计算模块

## 概述

本模块实现了基于 SAM3 模型的细粒度奖励计算，用于为不同类别的图像生成任务提供更精确的奖励信号。

## 功能特性

1. **元数据加载**：从 `geneval_and_t2i_data_final.jsonl` 中加载元数据，按 prompt 快速查找  
2. **SAM3 检测**：单对象逐个 prompt，默认置信度阈值 0.5（与 SAM3Processor 一致）  
3. **多类别得分计算**（均使用匹配名称的最高置信度框）：
   - **spatial**：对象置信度 + 位置关系
   - **numeracy**：存在置信度 / 数量精准各 50%
   - **object / color / shape / texture / non**：存在覆盖率 × 平均置信度
   - **complex**：有空间信息则直接用 spatial 得分，否则用 object 得分

## 文件结构

```
sam3_deps/
├── __init__.py              # 模块初始化
├── metadata_loader.py       # 元数据加载器
├── sam3_detector.py         # SAM3 检测器封装
├── score_calculator.py      # 得分计算器
├── sam3_reward.py           # SAM3 奖励计算主模块
├── merge_metadata.py        # 元数据合并脚本
└── README.md                # 本文档
```

## 使用方法

### 1. 合并元数据到训练数据（可选）

在计算奖励之前，需要先将元数据合并到训练数据的 `ground_truth` 中：

```bash
python merge_metadata.py \
    --train_data /path/to/train.json \
    --metadata_jsonl /path/to/geneval_and_t2i_data_final.jsonl \
    --output /path/to/train_with_metadata.json
```

### 2. 在奖励计算中使用 SAM3 奖励

SAM3 奖励已集成到 `examples/reward_function/self_reward_staged_reward_api.py` 的 `compute_stage2_reward_api` 函数中。

默认情况下，SAM3 奖励是启用的。如果编辑成功，系统会：
1. 使用 CLIP/self_reward 计算基础奖励
2. 使用 SAM3 计算细粒度奖励
3. 将两者结合：`最终奖励 = 0.5 * 基础奖励 + 0.5 * SAM3 奖励`

### 3. 直接使用 SAM3 奖励计算

```python
from sam3_deps.sam3_reward import compute_sam3_reward

result = compute_sam3_reward(
    image_path="/path/to/image.png",
    prompt="a photo of a car below a zebra",
    category="spatial",
    ground_truth='{"answer": true, "category": "spatial", ...}',
    device="cuda"
)

print(f"得分: {result['score']}")
print(f"成功: {result['success']}")
```

## 得分计算逻辑

### Spatial（空间关系）

1. 检测 obj1/obj2，分别取最高置信度框  
2. 位置得分按分辨率自适应距离阈值  
3. 综合得分 = 0.25 * obj1_conf + 0.25 * obj2_conf + 0.5 * 位置得分  
   若仅匹配到一个对象：返回 0.25 * 该对象置信度

### Numeracy（数量）

1. 检测期望对象，收集匹配置信度  
2. 存在置信度均值与“数量精准匹配”各占 50%  
3. 综合得分 = weight * (0.5 * mean_conf + 0.5 * count_exact)，weight=1/对象种类数

### Object / Color / Shape / Texture / Non

得分 = (检测覆盖率) × (匹配到的平均置信度)，覆盖率 = 命中数量 / 期望数量。

### Complex（复杂任务）

- 有 `spatial_info`：直接返回 spatial 得分  
- 无 `spatial_info`：返回对象存在得分

## 注意事项

1. **SAM3 模型加载**：首次加载可能耗时，可通过 `SAM3Detector(confidence_threshold=...)` 调整阈值。  
2. **设备要求**：建议 GPU（cuda），CPU 较慢。  
3. **元数据匹配**：找不到 prompt 对应元数据会返回 0 分。  
4. **检测失败**：不影响主流程，只返回基础奖励。  
5. **单对象 prompt**：检测逐对象提示，利于计数与属性对齐。

## 依赖项

- SAM3 模型（需要安装 sam3 包）
- PIL/Pillow
- torch
- 其他依赖见 `requirements.txt`

## 配置

可以在 `compute_stage2_reward_api` 函数中调整 SAM3 奖励的权重：

```python
# 当前权重：0.5 * 基础奖励 + 0.5 * SAM3 奖励
score = 0.5 * score + 0.5 * sam3_score
```
