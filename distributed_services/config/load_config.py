#!/usr/bin/env python3
"""
从 YAML 配置文件加载配置并导出为环境变量

用途：将 config.yaml 转换为环境变量，供 bash 脚本使用

使用方式：
    source "$(python3 config/load_config.py)"
    或
    eval "$(python3 config/load_config.py)"
"""

import sys
import os
import yaml
import argparse
import subprocess

def detect_gpu_count():
    """检测可用的GPU数量"""
    # 方法1: 使用 nvidia-smi（最简单可靠）
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
            if gpu_count > 0:
                return gpu_count
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # 方法2: 使用 PyTorch（如果可用）
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                return gpu_count
    except ImportError:
        pass
    
    # 方法3: 检查 CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # 解析 CUDA_VISIBLE_DEVICES，例如 "0,1,2,3" 或 "0-3"
        if "," in cuda_visible:
            return len([x.strip() for x in cuda_visible.split(",") if x.strip()])
        elif "-" in cuda_visible:
            parts = cuda_visible.split("-")
            if len(parts) == 2:
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    return end - start + 1
                except ValueError:
                    pass
    
    # 默认返回 0（表示未检测到GPU）
    return 0


def get_env_vars(config):
    """从 YAML 配置中提取环境变量"""
    env_vars = {}
    
    # 服务配置
    service = config.get("service", {})
    
    # 服务端口配置
    env_vars["EDIT_SERVER_BASE_PORT"] = str(service.get("edit_server_base_port", 5001))
    env_vars["REWARD_SERVER_BASE_PORT"] = str(service.get("reward_server_base_port", 5002))
    env_vars["SAM3_REWARD_SERVER_BASE_PORT"] = str(service.get("sam3_reward_server_base_port", 5100))
    
    # 模型路径配置
    env_vars["EDIT_MODEL_PATH"] = service.get("edit_model_path", "")
    env_vars["SELF_REWARD_MODEL_PATH"] = service.get("self_reward_model_path", "")
    env_vars["SELF_REWARD_MODEL_TYPE"] = service.get("self_reward_model_type", "omniverifier")
    
    # SAM3 模型资源路径
    env_vars["SAM3_BPE_PATH"] = service.get("sam3_bpe_path", "")
    env_vars["SAM3_CKPT_PATH"] = service.get("sam3_ckpt_path", "")
    env_vars["SAM3_METADATA_JSONL"] = service.get("sam3_metadata_jsonl", "")

    # 奖励服务配置
    env_vars["REWARD_TYPE"] = service.get("reward_type", "self_reward")
    env_vars["REWARD_TYPE_PER_GPU"] = service.get("reward_type_per_gpu", "")

    # GPU配置
    gpus_per_node = service.get("gpus_per_node", 0)
    if gpus_per_node == 0:
        # 如果配置为0，自动检测GPU数量
        detected_gpus = detect_gpu_count()
        env_vars["GPUS_PER_NODE"] = str(detected_gpus) if detected_gpus > 0 else "0"
    else:
        env_vars["GPUS_PER_NODE"] = str(gpus_per_node)
    # 统一的每GPU实例数配置（支持按GPU设置）
    instances_per_gpu = service.get("instances_per_gpu", "1")
    # 如果是数字，转换为字符串；如果是字符串（按GPU指定），直接使用
    env_vars["INSTANCES_PER_GPU"] = str(instances_per_gpu) if isinstance(instances_per_gpu, (int, float)) else instances_per_gpu
    
    # 图像编辑模型配置
    env_vars["EDIT_MODEL_TYPE"] = service.get("edit_model_type", "bagel")
    
    # CLIP配置
    env_vars["CLIP_MODULE_PATH"] = service.get("clip_module_path", "")
    env_vars["CLIP_MODEL_PATH"] = service.get("clip_model_path", "")
    
    # API配置
    env_vars["USE_API_MODE"] = str(service.get("use_api_mode", True)).lower()
    env_vars["EDIT_SERVER_THREADED"] = str(service.get("edit_server_threaded", True)).lower()
    
    # API客户端配置
    api_client = config.get("api_client", {})
    env_vars["API_ENABLE_HEALTH_CHECK"] = str(api_client.get("enable_health_check", True)).lower()
    env_vars["API_HEALTH_CHECK_FAILURE_THRESHOLD"] = str(api_client.get("health_check_failure_threshold", 5))
    env_vars["API_REQUEST_TIMEOUT"] = str(api_client.get("request_timeout", 180))
    env_vars["API_MAX_RETRIES"] = str(api_client.get("max_retries", 3))
    
    # 平台配置
    platform = config.get("platform", {})
    env_vars["SENSECORE_PYTORCH_NODE_RANK"] = str(platform.get("sensecore_pytorch_node_rank", 0))
    env_vars["SENSECORE_PYTORCH_NNODES"] = str(platform.get("sensecore_pytorch_nnodes", 8))
    env_vars["SENSECORE_ACCELERATE_DEVICE_COUNT"] = str(platform.get("sensecore_accelerate_device_count", 8))
    env_vars["MASTER_ADDR"] = platform.get("master_addr", "127.0.0.1")
    env_vars["MASTER_PORT"] = str(platform.get("master_port", 23456))
    env_vars["RAY_HEAD_PORT"] = str(platform.get("ray_head_port", 7001))
    
    
    # 其他配置
    other = config.get("other", {})
    hf_endpoint = other.get("hf_endpoint", "")
    if hf_endpoint:
        env_vars["HF_ENDPOINT"] = hf_endpoint
    wandb = other.get("wandb", {})
    env_vars["WANDB_API_KEY"] = wandb.get("api_key", "")
    env_vars["WANDB_MODE"] = wandb.get("mode", "offline")
    env_vars["PYTHONUNBUFFERED"] = str(other.get("pythonunbuffered", 1))
    env_vars["EDIT_CONDA_ENV"] = other.get("edit_conda_env", "bagelfast")
    env_vars["REWARD_CONDA_ENV"] = other.get("reward_conda_env", "sam3")
    
    # 路径配置（从 YAML 加载）s
    env_vars["PROJECT_ROOT"] = other.get("project_root", "")
    
    # 根据从 YAML 加载的路径配置自动生成派生变量
    # 注意：这些变量在 YAML 中不直接配置，而是根据路径自动计算
    project_root = env_vars.get("PROJECT_ROOT", "")
    distributed_services = os.path.join(project_root, "distributed_services")
    logs = os.path.join(distributed_services, "logs")
    config = os.path.join(distributed_services, "config")
    env_vars["LOG_DIR"] = logs
    env_vars["CONFIG_DIR"] = config
    
    return env_vars


def main():
    parser = argparse.ArgumentParser(description="从 YAML 配置文件加载配置并导出为环境变量")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（默认：脚本所在目录的 config.yaml）"
    )
    args = parser.parse_args()
    
    # 确定配置文件路径
    if args.config:
        config_file = args.config
    else:
        # 默认：脚本所在目录的 config.yaml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, "config.yaml")
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        print(f"# [ERROR] YAML 配置文件不存在: {config_file}", file=sys.stderr)
        sys.exit(1)
    
    # 加载 YAML 配置
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"# [ERROR] 无法加载 YAML 配置: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 获取环境变量
    env_vars = get_env_vars(config)
    
    # 输出 export 语句（只输出未设置的环境变量，保持优先级）
    for key, value in env_vars.items():
        # 如果环境变量已设置，跳过（保持用户设置的优先级）
        if key not in os.environ:
            # 转义特殊字符（bash 需要）
            value_str = str(value)
            value_escaped = value_str.replace('\\', '\\\\').replace('"', '\\"')
            print(f"export {key}=\"{value_escaped}\"")

    # 额外加载 service_endpoints.env 文件（如果存在）
    # 这确保了服务端点配置也被加载到环境变量中
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(script_dir, "service_endpoints.env")

    if os.path.exists(env_file):
        print(f"# [INFO] 加载服务端点配置文件: {env_file}", file=sys.stderr)
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # 跳过注释和空行
                    if not line or line.startswith('#'):
                        continue

                    # 只处理 export 语句
                    if not line.startswith('export '):
                        continue

                    # 解析 export KEY="VALUE" 或 export KEY=VALUE
                    export_part = line[7:].strip()  # 去掉 'export '
                    if '=' not in export_part:
                        continue

                    # 找到第一个 = 的位置（作为分隔符）
                    eq_pos = export_part.find('=')
                    key = export_part[:eq_pos].strip()
                    value_part = export_part[eq_pos+1:].strip()

                    # 处理引号包裹的值
                    if value_part.startswith('"') and value_part.endswith('"'):
                        # 双引号：去掉引号，保留内部内容
                        value = value_part[1:-1]
                    elif value_part.startswith("'") and value_part.endswith("'"):
                        # 单引号：去掉引号，保留内部内容
                        value = value_part[1:-1]
                    else:
                        # 无引号：直接使用，但需要去掉可能的注释
                        value = value_part
                        # 如果值中包含 #，且不在引号内，则去掉注释部分
                        if '#' in value:
                            # 简单处理：如果 # 不在引号内，则去掉注释
                            comment_pos = value.find('#')
                            if comment_pos >= 0:
                                value = value[:comment_pos].strip()

                    # 只输出未设置的环境变量（保持优先级）
                    if key and key not in os.environ:
                        value_escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                        print(f"export {key}=\"{value_escaped}\"")

        except Exception as e:
            print(f"# [WARNING] 无法加载服务端点配置: {e}", file=sys.stderr)
    else:
        print(f"# [WARNING] 服务端点配置文件不存在: {env_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
