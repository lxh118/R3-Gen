#!/bin/bash
# 多节点分布式服务部署脚本

# Usage: 
#   在编辑服务节点上运行：./deploy_services.sh edit_server
#   在奖励服务节点上运行：./deploy_services.sh reward_server
#   在训练节点上获取服务地址：./deploy_services.sh get_config

set -euo pipefail

# 获取脚本所在目录（用于计算 DISTRIBUTED_SERVICES_ROOT 以加载配置）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${DISTRIBUTED_SERVICES_ROOT:-}" ]]; then
    DISTRIBUTED_SERVICES_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

REPO_ROOT="$(cd "${DISTRIBUTED_SERVICES_ROOT}/.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
export LOG_DIR="${LOG_DIR:-${DISTRIBUTED_SERVICES_ROOT}/logs}"
export CONFIG_DIR="${CONFIG_DIR:-${DISTRIBUTED_SERVICES_ROOT}/config}"

SERVICE_TYPE=${1:-"help"}

# 所有配置变量（包括路径）都会从 YAML 加载并 export。help 不依赖
# PyYAML，因此允许用户在安装依赖前查看用法。
if [[ "$SERVICE_TYPE" != "help" && "$SERVICE_TYPE" != "--help" && "$SERVICE_TYPE" != "-h" ]]; then
    CONFIG_PYTHON="${CONFIG_PYTHON:-}"
    if [[ -z "$CONFIG_PYTHON" ]]; then
        for candidate in python3 python; do
            if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import yaml" >/dev/null 2>&1; then
                CONFIG_PYTHON="$candidate"
                break
            fi
        done
    fi
    CONFIG_PYTHON="${CONFIG_PYTHON:-python3}"

    if ! config_exports="$("$CONFIG_PYTHON" "${DISTRIBUTED_SERVICES_ROOT}/config/load_config.py")"; then
        echo "[ERROR] Failed to load config with CONFIG_PYTHON=${CONFIG_PYTHON}." >&2
        echo "[ERROR] Install PyYAML in that environment or run with CONFIG_PYTHON=/path/to/python." >&2
        exit 1
    fi
    eval "${config_exports}"
fi

# 创建必要的目录（使用从 YAML 加载的变量）
mkdir -p "${LOG_DIR:-${DISTRIBUTED_SERVICES_ROOT}/logs}"

# ============================================================================
# 脚本内部使用的局部变量（不从 YAML 加载，由脚本运行时设置）
# ============================================================================

# 节点地址配置（用于多节点部署，默认使用当前节点 hostname）
EDIT_SERVER_NODE="${EDIT_SERVER_NODE:-$(hostname)}"
REWARD_SERVER_NODE="${REWARD_SERVER_NODE:-$(hostname)}"

# ============================================================================
# 常量定义
# ============================================================================
readonly MAX_EDIT_INSTANCES_PER_GPU=2
readonly MAX_REWARD_INSTANCES_PER_GPU=32
readonly SERVICE_KEEPALIVE_SECONDS=259200  # 72小时

# ============================================================================
# 工具函数
# ============================================================================

# 获取节点标识符（用于日志文件名）
get_node_id() {
    local node_id=$(hostname | sed 's/[^a-zA-Z0-9]/_/g' | head -c 20)
    if [[ -z "$node_id" ]]; then
        node_id=$(hostname -I 2>/dev/null | awk '{print $1}' | tr -d '[:space:]' | sed 's/[^a-zA-Z0-9]/_/g' | head -c 20)
    fi
    echo "$node_id"
}

# 获取服务器IP地址
get_server_address() {
    local server_addr=$(hostname -I 2>/dev/null | awk '{print $1}' | tr -d '[:space:]')
    if [[ -z "$server_addr" ]]; then
        server_addr=$(hostname)
        echo "[WARNING] 无法获取IP地址，使用主机名: $server_addr" >&2
    fi
    echo "$server_addr"
}

# 获取Conda环境的Python命令
get_python_cmd() {
    local conda_env="${1:-}"
    local python_cmd="python3"
    
    if [[ -n "$conda_env" ]] && command -v conda &> /dev/null; then
        if conda env list | grep -q "^${conda_env} "; then
            local conda_base=$(conda info --base)
            python_cmd="${conda_base}/envs/${conda_env}/bin/python3"
            if [[ ! -f "$python_cmd" ]]; then
                python_cmd="python3"  # 回退到系统python
            fi
        fi
    fi
    
    echo "$python_cmd"
}

# 解析实例信息字符串到数组 (格式: gpu_id:instance_id:port:log_file[:reward_type])
# 参数: $1=实例信息字符串, $2=数组名（通过 nameref 返回）
parse_instance_info() {
    local info="$1"
    local -n parts_ref=$2
    IFS=':' read -ra parts_ref <<< "$info"
}

normalize_reward_type() {
    local reward_type
    reward_type="$(echo "${1:-self_reward}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "$reward_type" in
        self_reward|clip|sam3|mixed)
            echo "$reward_type"
            ;;
        *)
            echo "$reward_type"
            ;;
    esac
}

is_self_reward_type() {
    local reward_type
    reward_type="$(normalize_reward_type "$1")"
    [[ "$reward_type" == "self_reward" ]]
}

normalize_self_reward_model_type() {
    local model_type
    model_type="$(echo "${1:-omniverifier}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "$model_type" in
        omni|omniverifier)
            echo "omniverifier"
            ;;
        qwen3|qwen3vl|qwen3_vl)
            echo "qwen3vl"
            ;;
        qwen2_5|qwen2_5vl|qwen2.5|qwen2.5vl|qwen25|qwen25vl)
            echo "qwen2_5vl"
            ;;
        *)
            echo "$model_type"
            ;;
    esac
}

normalize_edit_model_type() {
    local model_type
    model_type="$(echo "${1:-bagel}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "$model_type" in
        qwen|qwen_image_edit)
            echo "qwen_image_edit"
            ;;
        bagel|bagel_edit)
            echo "bagel"
            ;;
        *)
            echo "$model_type"
            ;;
    esac
}

# 计算总实例数
calculate_total_instances() {
    local num_gpus=$1
    local -n gpu_instances_ref=$2
    local total=0
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        total=$((total + ${gpu_instances_ref[$gpu_id]}))
    done
    echo "$total"
}

# 检查服务状态并打印信息（通用函数）
# 参数: $1=服务名称, $2=加载提示信息, $3=服务器节点地址, $4=进程ID数组名, $5=实例信息数组名
check_and_print_service_status() {
    local service_name="$1"
    local loading_msg="$2"
    local server_node="$3"
    local -n pids_ref=$4
    local -n instance_info_ref=$5
    
    echo ""
    echo "⏳ 服务进程已启动，正在加载模型..."
    echo "   $loading_msg"
    echo ""
    echo "📋 实时查看加载进度："
    for info in "${instance_info_ref[@]}"; do
        local parts
        parse_instance_info "$info" "parts"
        local gpu_id="${parts[0]}"
        local instance_id="${parts[1]}"
        local log_file="${parts[3]}"
        local reward_type="${parts[4]:-}"
        if [[ -n "$reward_type" ]]; then
            echo "   tail -f ${log_file}  # GPU $gpu_id, 实例 $instance_id ($reward_type)"
        else
            echo "   tail -f ${log_file}  # GPU $gpu_id, 实例 $instance_id"
        fi
    done
    echo ""
    
    # 检查进程状态
    echo "🔍 检查服务启动状态..."
    local all_running=true
    for i in "${!pids_ref[@]}"; do
        local pid=${pids_ref[$i]}
        local info="${instance_info_ref[$i]}"
        local parts
        parse_instance_info "$info" "parts"
        local gpu_id="${parts[0]}"
        local instance_id="${parts[1]}"
        local log_file="${parts[3]}"
        
        sleep 3
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  ❌ GPU $gpu_id 实例 $instance_id 上的服务进程已退出（PID: $pid）"
            echo "     请检查日志: $log_file"
            echo "     最后几行日志:"
            tail -10 "$log_file" 2>/dev/null | sed 's/^/     /' || echo "     (日志文件为空或不存在)"
            all_running=false
        else
            echo "  ✅ GPU $gpu_id 实例 $instance_id 上的服务进程运行中（PID: $pid）"
        fi
    done
    
    if [[ "$all_running" == "false" ]]; then
        echo ""
        echo "⚠️  部分服务启动失败，但其他服务将继续运行"
        echo "💡 提示：请检查失败的GPU日志，成功的服务仍可正常使用"
        # 不再 exit 1，让成功的服务继续运行
        # exit 1
    fi
    
    echo ""
    echo "✅ 所有服务进程运行正常，模型正在加载中..."
    echo ""
    echo "服务地址列表（模型加载完成后可用）："
    for info in "${instance_info_ref[@]}"; do
        local parts
        parse_instance_info "$info" "parts"
        local gpu_id="${parts[0]}"
        local instance_id="${parts[1]}"
        local port="${parts[2]}"
        local reward_type="${parts[4]:-}"
        if [[ -n "$reward_type" ]]; then
            echo "  - http://${server_node}:$port  (GPU $gpu_id, 实例 $instance_id, $reward_type)"
        else
            echo "  - http://${server_node}:$port  (GPU $gpu_id, 实例 $instance_id)"
        fi
    done
    echo ""
    echo "💡 提示：模型加载完成后，服务会自动开始接受请求"
    echo "   可以通过以下方式确认服务就绪："
    if [[ ${#instance_info_ref[@]} -gt 0 ]]; then
        local first_parts
        parse_instance_info "${instance_info_ref[0]}" "first_parts"
        local first_port="${first_parts[2]}"
        echo "   curl http://${server_node}:$first_port/health"
    fi
}

# ============================================================================
# 配置解析函数
# ============================================================================

# 解析每个GPU的实例数量配置
# 支持格式：
#   - 单个数字：所有GPU使用相同实例数，如 "2"
#   - 按GPU指定：格式 "0:2,1:2,2:1,3:1" 表示GPU0和1各2个实例，GPU2和3各1个实例
parse_instances_per_gpu() {
    local instances_config="${INSTANCES_PER_GPU:-1}"
    local num_gpus=$1
    declare -gA gpu_instances  # 全局关联数组
    
    # 如果配置是单个数字，所有GPU使用相同实例数
    if [[ "$instances_config" =~ ^[0-9]+$ ]]; then
        local instances_per_gpu=$instances_config
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            gpu_instances[$gpu_id]=$instances_per_gpu
        done
        echo "每个GPU启动 $instances_per_gpu 个实例"
    else
        # 解析按GPU指定的配置
        IFS=',' read -ra entries <<< "$instances_config"
        for entry in "${entries[@]}"; do
            IFS=':' read -ra parts <<< "$entry"
            if [[ ${#parts[@]} -eq 2 ]]; then
                local gpu_idx="${parts[0]}"
                local instances="${parts[1]}"
                if [[ "$gpu_idx" -ge 0 && "$gpu_idx" -lt "$num_gpus" ]]; then
                    gpu_instances["$gpu_idx"]="$instances"
                fi
            fi
        done
        
        # 对于未指定的GPU，默认使用1个实例
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            if [[ -z "${gpu_instances[$gpu_id]:-}" ]]; then
                gpu_instances[$gpu_id]=1
            fi
        done
        
        echo "按GPU指定实例数:"
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            echo "  GPU $gpu_id: ${gpu_instances[$gpu_id]} 个实例"
        done
    fi
}

# 启动图像编辑服务（BAGEL 或 Qwen-Image-Edit）
start_edit_server() {
    echo "========================================="
    echo "启动图像编辑服务"
    echo "========================================="
    
    local model_path="${1:-${EDIT_MODEL_PATH:-/path/to/BAGEL-7B-MoT}}"
    local edit_model_type
    edit_model_type="$(normalize_edit_model_type "${2:-${EDIT_MODEL_TYPE:-bagel}}")"
    
    # 获取GPU数量（从YAML配置加载，load_config.py已自动检测）
    local num_gpus="${GPUS_PER_NODE:-0}"
    
    echo "使用 $num_gpus 个GPU（从YAML配置或自动检测）"
    
    # NOTE: bagel只开启一个服务实例
    INSTANCES_PER_GPU=1

    # 解析每个GPU的实例数量配置
    parse_instances_per_gpu "$num_gpus"
    
    echo "模型路径: $model_path"
    echo "图像编辑模型类型: $edit_model_type"
    
    # 计算总实例数
    local total_instances=$(calculate_total_instances "$num_gpus" "gpu_instances")
    echo "总共将启动 $total_instances 个服务实例"
    
    # 获取Python命令
    local python_cmd=$(get_python_cmd "${EDIT_CONDA_ENV:-bagelfast}")
    local node_id=$(get_node_id)
    
    # 启动服务实例
    local pids=()
    local instance_info=()
    
    # 根据model_type选择服务器脚本
    local server_script=""
    case "$edit_model_type" in
        qwen_image_edit)
            server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/qwen_image_edit_server.py"
            ;;
        bagel)
            server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/bagel_edit_server.py"
            ;;
        *)
            echo "[ERROR] 未知的图像编辑模型类型: $edit_model_type"
            echo "支持的 EDIT_MODEL_TYPE: bagel, qwen_image_edit"
            exit 1
            ;;
    esac
    
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        local instances_per_gpu=${gpu_instances[$gpu_id]}
        for instance_id in $(seq 0 $((instances_per_gpu - 1))); do
            local port=$((EDIT_SERVER_BASE_PORT + gpu_id * MAX_EDIT_INSTANCES_PER_GPU + instance_id))
            local log_file="${LOG_DIR}/edit_server_${node_id}_gpu${gpu_id}_inst${instance_id}.log"
            
            echo "启动 GPU $gpu_id 上的实例 $instance_id/$((instances_per_gpu - 1))，端口: $port"
            
            CUDA_VISIBLE_DEVICES=$gpu_id nohup "$python_cmd" -u "$server_script" \
                --model_path "$model_path" \
                --port "$port" \
                --host "0.0.0.0" \
                > "$log_file" 2>&1 &

            pids+=($!)
            instance_info+=("$gpu_id:$instance_id:$port:$log_file")
            echo "  PID: ${pids[-1]}, 日志: $log_file"
            sleep 3
        done
        sleep 3
    done
    
    # 检查服务状态并打印信息
    check_and_print_service_status "image-edit" "图像编辑模型较大，加载可能需要2-5分钟，请耐心等待" \
        "$EDIT_SERVER_NODE" "pids" "instance_info"
    
    # 生成配置文件供其他节点使用（传递总实例数）
    generate_edit_server_config "$total_instances"
}

# 启动奖励服务（支持 CLIP、R3-Gen self-reward、SAM3、mixed，支持按GPU指定类型）
start_reward_server() {
    echo "========================================="
    echo "启动奖励服务"
    echo "========================================="

    # 获取GPU数量（从YAML配置加载，load_config.py已自动检测）
    local num_gpus="${GPUS_PER_NODE:-0}"

    if [[ "$num_gpus" -le 0 ]]; then
        echo "[ERROR] 未检测到可用GPU" >&2
        echo "[ERROR] 请检查：" >&2
        echo "  1. 在 config.yaml 的 other.gpus_per_node 中设置GPU数量" >&2
        echo "  2. 或确保 nvidia-smi 可用且能检测到GPU" >&2
        exit 1
    fi

    echo "使用 $num_gpus 个GPU（从YAML配置或自动检测）"
    
    local reward_type
    reward_type="$(normalize_reward_type "${REWARD_TYPE:-self_reward}")"
    local reward_type_per_gpu="${REWARD_TYPE_PER_GPU:-}"
    
    # 解析按GPU指定的奖励类型
    declare -A gpu_reward_types
    if [[ -n "$reward_type_per_gpu" ]]; then
        echo "检测到按GPU指定奖励类型: $reward_type_per_gpu"
        IFS=',' read -ra entries <<< "$reward_type_per_gpu"
        for entry in "${entries[@]}"; do
            IFS=':' read -ra parts <<< "$entry"
            if [[ ${#parts[@]} -eq 2 ]]; then
                local gpu_idx="${parts[0]}"
                local gpu_reward_type
                gpu_reward_type="$(normalize_reward_type "${parts[1]}")"
                gpu_reward_types["$gpu_idx"]="$gpu_reward_type"
                echo "  GPU $gpu_idx: $gpu_reward_type"
            fi
        done
    fi
    
    # 如果没有指定按GPU的奖励类型，或解析后没有任何有效的GPU配置，则为所有GPU使用默认的reward_type
    local has_any_gpu_config=false
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        if [[ -n "${gpu_reward_types[$gpu_id]:-}" ]]; then
            has_any_gpu_config=true
            break
        fi
    done

    if [[ "$has_any_gpu_config" == "false" ]]; then
        echo "未检测到按GPU指定的奖励类型，为所有GPU使用默认类型: $reward_type"
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            gpu_reward_types["$gpu_id"]="$reward_type"
            echo "  GPU $gpu_id: $reward_type"
        done
    fi
    
    # 准备服务参数
    local sam3_bpe_path="${SAM3_BPE_PATH:-}"
    local sam3_ckpt_path="${SAM3_CKPT_PATH:-}"
    local sam3_metadata_jsonl="${SAM3_METADATA_JSONL:-}"
    local self_reward_model_path="${SELF_REWARD_MODEL_PATH:-${MODEL_PATH:-}}"
    local self_reward_model_type
    self_reward_model_type="$(normalize_self_reward_model_type "${SELF_REWARD_MODEL_TYPE:-omniverifier}")"
    local clip_model_name="${1:-ViT-B/32}"  # 第一个参数作为CLIP模型名称
    
    # 验证必要的参数
    if [[ -n "$reward_type_per_gpu" ]]; then
        # 混合模式：检查每种类型所需的参数
        local has_sam3=false
        local has_clip=false
        local has_self_reward=false
        
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            local gpu_reward_type="${gpu_reward_types[$gpu_id]:-$reward_type}"
            if [[ "$gpu_reward_type" == "sam3" ]]; then
                has_sam3=true
            elif [[ "$gpu_reward_type" == "clip" ]]; then
                has_clip=true
            elif is_self_reward_type "$gpu_reward_type"; then
                has_self_reward=true
            fi
        done
        
        if [[ "$has_sam3" == "true" ]]; then
            if [[ -z "$sam3_bpe_path" || -z "$sam3_ckpt_path" ]]; then
                echo "[ERROR] SAM3服务需要设置 SAM3_BPE_PATH 和 SAM3_CKPT_PATH"
                exit 1
            fi
        fi
        
        if [[ "$has_self_reward" == "true" ]]; then
            if [[ -z "$self_reward_model_path" || ! -d "$self_reward_model_path" ]]; then
                echo "[ERROR] R3-Gen self-reward 模型路径不存在或未设置: $self_reward_model_path"
                echo "请设置 SELF_REWARD_MODEL_PATH，或在 config.yaml 的 service.self_reward_model_path 中设置路径"
                exit 1
            fi
            case "$self_reward_model_type" in
                omniverifier|qwen2_5vl|qwen3vl) ;;
                *)
                    echo "[ERROR] 未知的 SELF_REWARD_MODEL_TYPE: $self_reward_model_type"
                    echo "支持的 SELF_REWARD_MODEL_TYPE: omniverifier, qwen2_5vl, qwen3vl"
                    exit 1
                    ;;
            esac
        fi
    else
        # 单一模式：根据reward_type验证参数
        if is_self_reward_type "$reward_type"; then
            if [[ -z "$self_reward_model_path" || ! -d "$self_reward_model_path" ]]; then
                echo "[ERROR] R3-Gen self-reward 模型路径不存在或未设置: $self_reward_model_path"
                echo "请设置 SELF_REWARD_MODEL_PATH，或在 config.yaml 的 service.self_reward_model_path 中设置路径"
                exit 1
            fi
            case "$self_reward_model_type" in
                omniverifier|qwen2_5vl|qwen3vl) ;;
                *)
                    echo "[ERROR] 未知的 SELF_REWARD_MODEL_TYPE: $self_reward_model_type"
                    echo "支持的 SELF_REWARD_MODEL_TYPE: omniverifier, qwen2_5vl, qwen3vl"
                    exit 1
                    ;;
            esac
        elif [[ "$reward_type" == "sam3" ]]; then
            if [[ -z "$sam3_bpe_path" || -z "$sam3_ckpt_path" ]]; then
                echo "[ERROR] SAM3服务需要设置 SAM3_BPE_PATH 和 SAM3_CKPT_PATH"
                exit 1
            fi
        fi
    fi
    
    # 解析每个GPU的实例数量配置
    parse_instances_per_gpu "$num_gpus"
    
    # 计算总实例数
    local total_instances=$(calculate_total_instances "$num_gpus" "gpu_instances")
    echo "总共将启动 $total_instances 个服务实例"
    
    mkdir -p "${LOG_DIR}"
    
    # 启动服务实例
    local pids=()
    local reward_instance_info=()
    local node_id=$(get_node_id)
    local python_cmd=$(get_python_cmd "${REWARD_CONDA_ENV:-sam3}")
    
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        local gpu_reward_type="${gpu_reward_types[$gpu_id]:-$reward_type}"
        local instances_per_gpu=${gpu_instances[$gpu_id]}
        
        # 根据奖励类型选择服务脚本和端口
        local server_script=""
        local base_port=""
        case "$gpu_reward_type" in
            sam3)
                server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/sam3_reward_server.py"
                base_port="${SAM3_REWARD_SERVER_BASE_PORT:-5100}"
                ;;
            self_reward)
                case "$self_reward_model_type" in
                    omniverifier)
                        server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/self_reward_omniverifier_server.py"
                        ;;
                    qwen2_5vl)
                        server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/self_reward_qwen2_5vl_server.py"
                        ;;
                    qwen3vl)
                        server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/self_reward_qwen3vl_server.py"
                        ;;
                    *)
                        echo "[ERROR] 未知的 SELF_REWARD_MODEL_TYPE: $self_reward_model_type"
                        exit 1
                        ;;
                esac
                base_port="${REWARD_SERVER_BASE_PORT:-5002}"
                ;;
            *)
                server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/clip_reward_server.py"
                base_port="${REWARD_SERVER_BASE_PORT:-5002}"
                ;;
        esac
        
        # 为当前GPU启动多个实例
        for instance_id in $(seq 0 $((instances_per_gpu - 1))); do
            local port=$((base_port + gpu_id * MAX_REWARD_INSTANCES_PER_GPU + instance_id))
            local log_suffix="$gpu_reward_type"
            if [[ "$gpu_reward_type" == "self_reward" ]]; then
                log_suffix="${gpu_reward_type}_${self_reward_model_type}"
            fi
            local log_file="${LOG_DIR}/reward_server_${node_id}_gpu${gpu_id}_inst${instance_id}_${log_suffix}.log"
            
            echo "启动 GPU $gpu_id 上的实例 $instance_id/$((instances_per_gpu - 1))（类型: $log_suffix），端口: $port"
            
            # 启动对应类型的服务
            case "$gpu_reward_type" in
                self_reward)
                    CUDA_VISIBLE_DEVICES=$gpu_id nohup "$python_cmd" -u "$server_script" \
                        --model_path "$self_reward_model_path" \
                        --port "$port" \
                        --host "0.0.0.0" \
                        --device "$gpu_id" \
                        > "$log_file" 2>&1 &
                    ;;
                sam3)
                    CUDA_VISIBLE_DEVICES=$gpu_id nohup "$python_cmd" -u "$server_script" \
                        --bpe_path "$sam3_bpe_path" \
                        --ckpt_path "$sam3_ckpt_path" \
                        --metadata_jsonl "$sam3_metadata_jsonl" \
                        --port "$port" \
                        --host "0.0.0.0" \
                        --device "$gpu_id" \
                        > "$log_file" 2>&1 &
                    ;;
                *)
                    # CLIP服务
                    local cmd_args=(
                        --model_name "$clip_model_name"
                        --port "$port"
                        --host "0.0.0.0"
                        --device "$gpu_id"
                    )
                    [[ -n "${CLIP_MODULE_PATH}" && -f "${CLIP_MODULE_PATH}" ]] && \
                        cmd_args+=(--clip_module_path "${CLIP_MODULE_PATH}")
                    [[ -n "${CLIP_MODEL_PATH}" && -f "${CLIP_MODEL_PATH}" ]] && \
                        cmd_args+=(--model_path "${CLIP_MODEL_PATH}")
                    
                    CUDA_VISIBLE_DEVICES=$gpu_id nohup "$python_cmd" -u "$server_script" \
                        "${cmd_args[@]}" \
                        > "$log_file" 2>&1 &
                    ;;
            esac
            
            pids+=($!)
            reward_instance_info+=("$gpu_id:$instance_id:$port:$log_file:$gpu_reward_type")
            echo "  PID: ${pids[-1]}, 日志: $log_file"
            sleep 1
        done
        sleep 1
    done
    
    # 检查服务状态并打印信息
    check_and_print_service_status "奖励服务" "模型加载可能需要几分钟，请耐心等待" \
        "$REWARD_SERVER_NODE" "pids" "reward_instance_info"
    
    # 生成配置文件供其他节点使用（传递总实例数）
    generate_reward_server_config "$total_instances" "${reward_instance_info[@]}"
}

# 读取现有端点到关联数组（用于去重）
read_existing_endpoints() {
    local config_file="$1"
    local -n endpoints_ref=$2
    if [[ -f "$config_file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            line=$(echo "$line" | tr -d '[:space:]')
            [[ -n "$line" ]] && endpoints_ref["$line"]=1
        done < "$config_file"
    fi
}

# 追加端点到配置文件（去重）
# 参数: $1=配置文件路径, $2=端点URL, $3=关联数组名（用于检查是否已存在）, $4=计数器变量名（用于统计）
append_endpoint_to_file() {
    local config_file="$1"
    local endpoint="$2"
    local existing_array_name="$3"
    local count_var_name="$4"
    
    # 使用 eval 检查端点是否已存在
    local exists=0
    eval "exists=\${${existing_array_name}[\"$endpoint\"]:-0}"
    
    if [[ $exists -eq 0 ]]; then
        echo "$endpoint" >> "$config_file"
        eval "${existing_array_name}[\"$endpoint\"]=1"
        eval "${count_var_name}=\$((\${${count_var_name}} + 1))"
        return 0
    else
        echo "[INFO] 端点已存在，跳过: $endpoint" >&2
        return 1
    fi
}

# 生成图像编辑服务配置文件
generate_edit_server_config() {
    local num_instances=$1
    local config_file="${CONFIG_DIR}/edit_server_endpoints.txt"
    
    mkdir -p "${CONFIG_DIR}"
    local server_addr=$(get_server_address)
    echo "[INFO] 使用IP地址: $server_addr"
    
    # 读取现有端点
    declare -A existing_endpoints
    read_existing_endpoints "$config_file" "existing_endpoints"
    
    local added_count=0
    
    # 如果存在全局 instance_info 数组，使用它
    if [[ -n "${instance_info[@]:-}" ]]; then
        for info in "${instance_info[@]}"; do
            local parts
            parse_instance_info "$info" "parts"
            local port="${parts[2]}"
            local endpoint="http://${server_addr}:$port"
            append_endpoint_to_file "$config_file" "$endpoint" "existing_endpoints" "added_count" >/dev/null
        done
    else
        # 每个GPU一个实例
        for gpu_id in $(seq 0 $((num_instances - 1))); do
            local endpoint="http://${server_addr}:$((EDIT_SERVER_BASE_PORT + gpu_id))"
            append_endpoint_to_file "$config_file" "$endpoint" "existing_endpoints" "added_count" >/dev/null
        done
    fi
    
    if [[ $added_count -gt 0 ]]; then
        echo "配置文件已更新: $config_file（追加了 $added_count 个新端点）"
    else
        echo "配置文件未更新: $config_file（所有端点已存在）"
    fi
}

# 生成奖励服务配置文件
generate_reward_server_config() {
    local num_instances=$1
    shift
    local reward_instance_info=("$@")
    
    local config_file="${CONFIG_DIR}/reward_server_endpoints.txt"
    local clip_config_file="${CONFIG_DIR}/clip_reward_server_endpoints.txt"
    local self_reward_config_file="${CONFIG_DIR}/self_reward_server_endpoints.txt"
    local sam3_config_file="${CONFIG_DIR}/sam3_reward_server_endpoints.txt"
    
    mkdir -p "${CONFIG_DIR}"
    local server_addr=$(get_server_address)
    echo "[INFO] 使用IP地址: $server_addr"
    
    # 读取现有端点（用于去重）
    declare -A existing_endpoints
    declare -A existing_clip_endpoints
    declare -A existing_self_reward_endpoints
    declare -A existing_sam3_endpoints
    
    read_existing_endpoints "$config_file" "existing_endpoints"
    read_existing_endpoints "$clip_config_file" "existing_clip_endpoints"
    read_existing_endpoints "$self_reward_config_file" "existing_self_reward_endpoints"
    read_existing_endpoints "$sam3_config_file" "existing_sam3_endpoints"
    
    # 追加端点
    local added_count=0
    local added_clip_count=0
    local added_self_reward_count=0
    local added_sam3_count=0
    
    if [[ ${#reward_instance_info[@]} -gt 0 ]]; then
        for info in "${reward_instance_info[@]}"; do
            local parts
            parse_instance_info "$info" "parts"
            local port="${parts[2]}"
            local reward_type="${parts[4]:-clip}"
            local endpoint="http://${server_addr}:$port"
            
            # 根据类型选择目标文件和关联数组名
            local target_file=""
            local existing_array_name="existing_clip_endpoints"
            local count_var_name="added_clip_count"
            
            case "$reward_type" in
                sam3)
                    target_file="$sam3_config_file"
                    existing_array_name="existing_sam3_endpoints"
                    count_var_name="added_sam3_count"
                    ;;
                self_reward)
                    target_file="$self_reward_config_file"
                    existing_array_name="existing_self_reward_endpoints"
                    count_var_name="added_self_reward_count"
                    ;;
                *)
                    target_file="$clip_config_file"
                    existing_array_name="existing_clip_endpoints"
                    count_var_name="added_clip_count"
                    ;;
            esac
            
            # 追加到分类配置文件
            if append_endpoint_to_file "$target_file" "$endpoint" "$existing_array_name" "$count_var_name"; then
                added_count=$((added_count + 1))
                # 同时写入统一配置文件
                append_endpoint_to_file "$config_file" "$endpoint" "existing_endpoints" "added_count" >/dev/null
            fi
        done
    else
        # 每个GPU一个实例
        local default_reward_type
        default_reward_type="$(normalize_reward_type "${REWARD_TYPE:-self_reward}")"
        local target_file=""
        local existing_array_name="existing_clip_endpoints"
        local count_var_name="added_clip_count"
        case "$default_reward_type" in
            sam3)
                target_file="$sam3_config_file"
                existing_array_name="existing_sam3_endpoints"
                count_var_name="added_sam3_count"
                ;;
            self_reward)
                target_file="$self_reward_config_file"
                existing_array_name="existing_self_reward_endpoints"
                count_var_name="added_self_reward_count"
                ;;
            *)
                target_file="$clip_config_file"
                ;;
        esac
        
        for gpu_id in $(seq 0 $((num_instances - 1))); do
            local endpoint="http://${server_addr}:$((REWARD_SERVER_BASE_PORT + gpu_id))"
            append_endpoint_to_file "$config_file" "$endpoint" "existing_endpoints" "added_count" >/dev/null
            append_endpoint_to_file "$target_file" "$endpoint" "$existing_array_name" "$count_var_name" >/dev/null
        done
    fi
    
    # 统计各类型端点总数
    local total_clip_count=0
    local total_self_reward_count=0
    local total_sam3_count=0
    for info in "${reward_instance_info[@]}"; do
        local parts
        parse_instance_info "$info" "parts"
        case "${parts[4]:-clip}" in
            sam3) total_sam3_count=$((total_sam3_count + 1)) ;;
            self_reward) total_self_reward_count=$((total_self_reward_count + 1)) ;;
            *) total_clip_count=$((total_clip_count + 1)) ;;
        esac
    done
    
    echo ""
    echo "========================================="
    echo "配置文件生成结果"
    echo "========================================="
    echo "当前节点IP: $server_addr"
    echo "配置文件路径:"
    echo "  - $config_file（统一配置文件）"
    echo "  - $clip_config_file（CLIP端点：当前节点 $total_clip_count 个实例）"
    echo "  - $self_reward_config_file（R3-Gen self-reward端点：当前节点 $total_self_reward_count 个实例）"
    echo "  - $sam3_config_file（SAM3端点：当前节点 $total_sam3_count 个实例）"
    
    if [[ $added_count -gt 0 ]]; then
        echo ""
        echo "✅ 配置文件已更新（追加了新端点）:"
        echo "  - $config_file（追加了 $added_count 个新端点）"
        if [[ $added_clip_count -gt 0 ]]; then
            echo "  - $clip_config_file（追加了 $added_clip_count 个 CLIP 端点）"
        fi
        if [[ $added_self_reward_count -gt 0 ]]; then
            echo "  - $self_reward_config_file（追加了 $added_self_reward_count 个 R3-Gen self-reward 端点）"
        fi
        if [[ $added_sam3_count -gt 0 ]]; then
            echo "  - $sam3_config_file（追加了 $added_sam3_count 个 SAM3 端点）"
        fi
    else
        echo ""
        echo "ℹ️  配置文件已检查（所有端点已存在，无需更新）"
        if [[ $total_clip_count -gt 0 ]]; then
            echo "  - CLIP端点: $total_clip_count 个实例已配置"
        fi
        if [[ $total_self_reward_count -gt 0 ]]; then
            echo "  - R3-Gen self-reward端点: $total_self_reward_count 个实例已配置"
        fi
        if [[ $total_sam3_count -gt 0 ]]; then
            echo "  - SAM3端点: $total_sam3_count 个实例已配置"
        fi
    fi
    echo "========================================="
    echo ""
}

# 停止图像编辑服务
stop_edit_server() {
    echo "停止图像编辑服务..."
    
    pkill -f "distributed_services/servers/bagel_edit_server.py" || pkill -f "bagel_edit_server.py" || true
    pkill -f "distributed_services/servers/qwen_image_edit_server.py" || pkill -f "qwen_image_edit_server.py" || true
    
    echo "✅ 图像编辑服务已停止"
    echo "💡 奖励服务仍在运行"
}

# 停止奖励服务（只停止奖励服务）
stop_reward_server() {
    echo "停止奖励服务..."
    
    # 停止奖励服务（CLIP / self-reward / SAM3）
    pkill -f "distributed_services/servers/clip_reward_server.py" || pkill -f "clip_reward_server.py" || true
    pkill -f "distributed_services/servers/self_reward_omniverifier_server.py" || pkill -f "self_reward_omniverifier_server.py" || true
    pkill -f "distributed_services/servers/self_reward_qwen2_5vl_server.py" || pkill -f "self_reward_qwen2_5vl_server.py" || true
    pkill -f "distributed_services/servers/self_reward_qwen3vl_server.py" || pkill -f "self_reward_qwen3vl_server.py" || true
    pkill -f "distributed_services/servers/sam3_reward_server.py" || pkill -f "sam3_reward_server.py" || true
    
    echo "✅ 奖励服务已停止"
    echo "💡 图像编辑服务仍在运行"
}

# 停止所有服务
stop_services() {
    echo "停止所有服务..."
    
    # 停止图像编辑服务
    pkill -f "distributed_services/servers/bagel_edit_server.py" || pkill -f "bagel_edit_server.py" || true
    pkill -f "distributed_services/servers/qwen_image_edit_server.py" || pkill -f "qwen_image_edit_server.py" || true
    
    # 停止CLIP奖励服务
    pkill -f "distributed_services/servers/clip_reward_server.py" || pkill -f "clip_reward_server.py" || true
    # 停止SAM3奖励服务
    pkill -f "distributed_services/servers/sam3_reward_server.py" || pkill -f "sam3_reward_server.py" || true
    # 停止self-reward奖励服务
    pkill -f "distributed_services/servers/self_reward_omniverifier_server.py" || pkill -f "self_reward_omniverifier_server.py" || true
    pkill -f "distributed_services/servers/self_reward_qwen2_5vl_server.py" || pkill -f "self_reward_qwen2_5vl_server.py" || true
    pkill -f "distributed_services/servers/self_reward_qwen3vl_server.py" || pkill -f "self_reward_qwen3vl_server.py" || true
    
    echo "✅ 所有服务已停止"
}
# ============================================================================
# 配置生成相关函数
# ============================================================================

# 从文件读取端点（去重）
# 参数: $1=配置文件路径, $2=关联数组引用（用于去重，通过 nameref）
# 返回: 逗号分隔的端点字符串
read_endpoints_from_file() {
    local config_file="$1"
    local -n seen_array="$2"  # nameref，直接引用传入的关联数组
    local endpoints=""
    
    if [[ ! -f "$config_file" ]]; then
        echo ""
        return
    fi
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        line=$(echo "$line" | tr -d '[:space:]')
        if [[ -n "$line" && "$line" =~ ^http:// ]]; then
            # 检查是否已存在（去重）
            if [[ -z "${seen_array[$line]:-}" ]]; then
                seen_array[$line]=1
                if [[ -n "$endpoints" ]]; then
                    endpoints="${endpoints},"
                fi
                endpoints="${endpoints}${line}"
            fi
        fi
    done < "$config_file"
    
    echo "$endpoints"
}

# 读取编辑服务端点
read_edit_server_endpoints() {
    local edit_config="${CONFIG_DIR}/edit_server_endpoints.txt"
    declare -A seen_edit_endpoints
    
    if [[ -f "$edit_config" ]]; then
        echo "[INFO] 从配置文件读取编辑服务端点: $edit_config" >&2
        local endpoints=$(read_endpoints_from_file "$edit_config" "seen_edit_endpoints")
        local count=$(echo "$endpoints" | tr ',' '\n' | grep -c '^http://' 2>/dev/null || echo "0")
        echo "[INFO] 读取到 $count 个编辑服务端点" >&2
        echo "$endpoints"
    else
        echo ""
    fi
}

# 读取奖励服务端点（分类读取）
# 返回: 通过全局变量 REWARD_ENDPOINTS_CLIP, REWARD_ENDPOINTS_SELF_REWARD, REWARD_ENDPOINTS_SAM3
read_reward_server_endpoints() {
    local clip_config_file="${CONFIG_DIR}/clip_reward_server_endpoints.txt"
    local self_reward_config_file="${CONFIG_DIR}/self_reward_server_endpoints.txt"
    local sam3_config_file="${CONFIG_DIR}/sam3_reward_server_endpoints.txt"
    local reward_config="${CONFIG_DIR}/reward_server_endpoints.txt"
    local sam3_base_port="${SAM3_REWARD_SERVER_BASE_PORT:-7001}"
    
    declare -A seen_clip_endpoints
    declare -A seen_self_reward_endpoints
    declare -A seen_sam3_endpoints
    
    local reward_endpoints_clip=""
    local reward_endpoints_self_reward=""
    local reward_endpoints_sam3=""
    local has_classified_configs=false
    
    # 优先从分类配置文件读取
    if [[ -f "$clip_config_file" ]]; then
        has_classified_configs=true
        echo "[INFO] 从分类配置文件读取 CLIP 服务端点: $clip_config_file"
        reward_endpoints_clip=$(read_endpoints_from_file "$clip_config_file" "seen_clip_endpoints")
    fi
    
    if [[ -f "$self_reward_config_file" ]]; then
        has_classified_configs=true
        echo "[INFO] 从分类配置文件读取 R3-Gen self-reward 服务端点: $self_reward_config_file"
        reward_endpoints_self_reward=$(read_endpoints_from_file "$self_reward_config_file" "seen_self_reward_endpoints")
    fi
    
    if [[ -f "$sam3_config_file" ]]; then
        has_classified_configs=true
        echo "[INFO] 从分类配置文件读取 SAM3 服务端点: $sam3_config_file"
        reward_endpoints_sam3=$(read_endpoints_from_file "$sam3_config_file" "seen_sam3_endpoints")
    fi
    
    # 如果分类配置文件不存在，从统一配置文件读取
    if [[ "$has_classified_configs" == "false" && -f "$reward_config" ]]; then
        echo "[INFO] 分类配置文件不存在，从统一配置文件读取奖励服务端点: $reward_config"
        while IFS= read -r line || [[ -n "$line" ]]; do
            line=$(echo "$line" | tr -d '[:space:]')
            if [[ -n "$line" && "$line" =~ ^http:// ]]; then
                # 根据端口判断是 SAM3 还是默认奖励端点
                if [[ "$line" =~ :([0-9]+)$ ]]; then
                    local port="${BASH_REMATCH[1]}"
                    if [[ $port -ge $sam3_base_port ]]; then
                        # SAM3 端点
                        if [[ -z "${seen_sam3_endpoints[$line]:-}" ]]; then
                            seen_sam3_endpoints[$line]=1
                            if [[ -n "$reward_endpoints_sam3" ]]; then
                                reward_endpoints_sam3="${reward_endpoints_sam3},"
                            fi
                            reward_endpoints_sam3="${reward_endpoints_sam3}${line}"
                        fi
                    else
                        # 默认作为当前 REWARD_TYPE 对应的端点
                        local default_reward_type
                        default_reward_type="$(normalize_reward_type "${REWARD_TYPE:-self_reward}")"
                        if [[ "$default_reward_type" == "self_reward" ]]; then
                            if [[ -z "${seen_self_reward_endpoints[$line]:-}" ]]; then
                                seen_self_reward_endpoints[$line]=1
                                if [[ -n "$reward_endpoints_self_reward" ]]; then
                                    reward_endpoints_self_reward="${reward_endpoints_self_reward},"
                                fi
                                reward_endpoints_self_reward="${reward_endpoints_self_reward}${line}"
                            fi
                            continue
                        fi
                        if [[ -z "${seen_clip_endpoints[$line]:-}" ]]; then
                            seen_clip_endpoints[$line]=1
                            if [[ -n "$reward_endpoints_clip" ]]; then
                                reward_endpoints_clip="${reward_endpoints_clip},"
                            fi
                            reward_endpoints_clip="${reward_endpoints_clip}${line}"
                        fi
                    fi
                else
                    # 无法解析端口，默认作为 CLIP 端点
                    if [[ -z "${seen_clip_endpoints[$line]:-}" ]]; then
                        seen_clip_endpoints[$line]=1
                        if [[ -n "$reward_endpoints_clip" ]]; then
                            reward_endpoints_clip="${reward_endpoints_clip},"
                        fi
                        reward_endpoints_clip="${reward_endpoints_clip}${line}"
                    fi
                fi
            fi
        done < "$reward_config"
    fi
    
    # 计算并输出各类型端点数量
    local clip_count=0
    local self_reward_count=0
    local sam3_count=0
    
    if [[ -n "$reward_endpoints_clip" ]]; then
        clip_count=$(echo "$reward_endpoints_clip" | tr ',' '\n' | grep -c '^http://' 2>/dev/null || echo "0")
    fi
    if [[ -n "$reward_endpoints_self_reward" ]]; then
        self_reward_count=$(echo "$reward_endpoints_self_reward" | tr ',' '\n' | grep -c '^http://' 2>/dev/null || echo "0")
    fi
    if [[ -n "$reward_endpoints_sam3" ]]; then
        sam3_count=$(echo "$reward_endpoints_sam3" | tr ',' '\n' | grep -c '^http://' 2>/dev/null || echo "0")
    fi
    
    echo "[INFO] 读取到 $clip_count 个 CLIP 服务端点"
    echo "[INFO] 读取到 $self_reward_count 个 R3-Gen self-reward 服务端点"
    echo "[INFO] 读取到 $sam3_count 个 SAM3 服务端点"
    
    # 返回端点值（通过全局变量）
    REWARD_ENDPOINTS_CLIP="$reward_endpoints_clip"
    REWARD_ENDPOINTS_SELF_REWARD="$reward_endpoints_self_reward"
    REWARD_ENDPOINTS_SAM3="$reward_endpoints_sam3"
}

# 生成默认端点
# 参数: $1=当前节点IP, $2=编辑服务端点变量名（通过 nameref 修改）, $3=CLIP端点变量名, $4=self-reward端点变量名, $5=SAM3端点变量名
generate_default_endpoints() {
    local current_ip="$1"
    local -n edit_endpoints_ref="$2"  # nameref，直接修改传入的变量
    local -n reward_endpoints_clip_ref="$3"
    local -n reward_endpoints_self_reward_ref="$4"
    local -n reward_endpoints_sam3_ref="$5"
    
    local num_gpus="${GPUS_PER_NODE:-8}"
    if [[ "$num_gpus" -le 0 ]]; then
        num_gpus=8  # 默认8个GPU
    fi
    
    # 生成编辑服务端点
    if [[ -z "$edit_endpoints_ref" ]]; then
        echo "[WARNING] edit_server_endpoints.txt 不存在或为空，使用当前节点生成编辑服务端点"
        for i in $(seq 0 $((num_gpus - 1))); do
            if [[ -n "$edit_endpoints_ref" ]]; then
                edit_endpoints_ref="${edit_endpoints_ref},"
            fi
            edit_endpoints_ref="${edit_endpoints_ref}http://${current_ip}:$((EDIT_SERVER_BASE_PORT + i))"
        done
        echo "[INFO] 生成了 $num_gpus 个编辑服务端点（基于当前节点: $current_ip）"
    fi
    
    # 生成奖励服务端点
    if [[ -z "$reward_endpoints_clip_ref" && -z "$reward_endpoints_self_reward_ref" && -z "$reward_endpoints_sam3_ref" ]]; then
        echo "[WARNING] reward_server_endpoints.txt 不存在或为空，使用当前节点生成奖励服务端点"
        local default_reward_type
        default_reward_type="$(normalize_reward_type "${REWARD_TYPE:-self_reward}")"
        
        if [[ "$default_reward_type" == "sam3" ]]; then
            # 生成 SAM3 端点
            for i in $(seq 0 $((num_gpus - 1))); do
                if [[ -n "$reward_endpoints_sam3_ref" ]]; then
                    reward_endpoints_sam3_ref="${reward_endpoints_sam3_ref},"
                fi
                reward_endpoints_sam3_ref="${reward_endpoints_sam3_ref}http://${current_ip}:$((SAM3_REWARD_SERVER_BASE_PORT + i))"
            done
            echo "[INFO] 生成了 $num_gpus 个 SAM3 奖励服务端点（基于当前节点: $current_ip）"
        elif [[ "$default_reward_type" == "self_reward" ]]; then
            # 生成 R3-Gen self-reward 端点
            for i in $(seq 0 $((num_gpus - 1))); do
                if [[ -n "$reward_endpoints_self_reward_ref" ]]; then
                    reward_endpoints_self_reward_ref="${reward_endpoints_self_reward_ref},"
                fi
                reward_endpoints_self_reward_ref="${reward_endpoints_self_reward_ref}http://${current_ip}:$((REWARD_SERVER_BASE_PORT + i))"
            done
            echo "[INFO] 生成了 $num_gpus 个 R3-Gen self-reward 奖励服务端点（基于当前节点: $current_ip）"
        else
            # 生成 CLIP 端点
            for i in $(seq 0 $((num_gpus - 1))); do
                if [[ -n "$reward_endpoints_clip_ref" ]]; then
                    reward_endpoints_clip_ref="${reward_endpoints_clip_ref},"
                fi
                reward_endpoints_clip_ref="${reward_endpoints_clip_ref}http://${current_ip}:$((REWARD_SERVER_BASE_PORT + i))"
            done
            echo "[INFO] 生成了 $num_gpus 个 CLIP 奖励服务端点（基于当前节点: $current_ip）"
        fi
    fi
}

# 根据 REWARD_TYPE 选择正确的端点
# 返回: 选中的端点字符串（通过 echo），更新的 reward_type（通过全局变量 UPDATED_REWARD_TYPE）
select_reward_endpoints() {
    local reward_type
    reward_type="$(normalize_reward_type "$1")"
    local reward_type_per_gpu="$2"
    local reward_endpoints_clip="$3"
    local reward_endpoints_self_reward="$4"
    local reward_endpoints_sam3="$5"
    
    local selected_reward_endpoints="${reward_endpoints_clip}"
    
    if [[ "$reward_type" == "self_reward" && -n "$reward_endpoints_self_reward" ]]; then
        selected_reward_endpoints="${reward_endpoints_self_reward}"
        echo "[INFO] REWARD_TYPE=self_reward，使用 R3-Gen self-reward 服务端点" >&2
    elif [[ "$reward_type" == "clip" && -n "$reward_endpoints_clip" ]]; then
        selected_reward_endpoints="${reward_endpoints_clip}"
        echo "[INFO] REWARD_TYPE=clip，使用 CLIP 服务端点" >&2
    elif [[ "$reward_type" == "sam3" && -n "$reward_endpoints_sam3" ]]; then
        selected_reward_endpoints=""
        echo "[INFO] REWARD_TYPE=sam3，使用 SAM3 服务端点" >&2
    elif [[ "$reward_type" == "mixed" ]]; then
        # mixed 模式：合并 CLIP 和 self-reward 端点
        selected_reward_endpoints=""
        for endpoints in "$reward_endpoints_clip" "$reward_endpoints_self_reward"; do
            if [[ -z "$endpoints" ]]; then
                continue
            fi
            if [[ -n "$selected_reward_endpoints" ]]; then
                selected_reward_endpoints="${selected_reward_endpoints},${endpoints}"
            else
                selected_reward_endpoints="${endpoints}"
            fi
        done
        echo "[INFO] REWARD_TYPE=mixed，合并 CLIP 和 self-reward 服务端点（通过 REWARD_TYPE_PER_GPU 区分）" >&2
    else
        echo "[WARNING] 未知的 REWARD_TYPE=$reward_type，使用 CLIP 端点（默认）" >&2
    fi
    
    # 如果检测到 SAM3 端点但 reward_type 不是 mixed，且没有设置 REWARD_TYPE_PER_GPU，则设置为 mixed
    if [[ -n "$reward_endpoints_sam3" && -n "$selected_reward_endpoints" && ( "$reward_type" == "clip" || "$reward_type" == "self_reward" ) && -z "$reward_type_per_gpu" ]]; then
        echo "[INFO] 检测到同时存在奖励服务和 SAM3 端点，将 REWARD_TYPE 设置为 mixed" >&2
        reward_type="mixed"
    fi
    
    # 通过全局变量返回更新后的 reward_type
    UPDATED_REWARD_TYPE="$reward_type"
    
    echo "$selected_reward_endpoints"
}

# 生成 service_endpoints.env 文件
generate_service_endpoints_env() {
    local export_file="$1"
    local edit_endpoints="$2"
    local selected_reward_endpoints="$3"
    local reward_endpoints_clip="$4"
    local reward_endpoints_self_reward="$5"
    local reward_endpoints_sam3="$6"
    local reward_type="$7"
    local reward_type_per_gpu="$8"
    
    # 确保值不为空（使用空字符串代替）
    edit_endpoints="${edit_endpoints:-}"
    selected_reward_endpoints="${selected_reward_endpoints:-}"
    reward_endpoints_clip="${reward_endpoints_clip:-}"
    reward_endpoints_self_reward="${reward_endpoints_self_reward:-}"
    reward_endpoints_sam3="${reward_endpoints_sam3:-}"
    reward_type="${reward_type:-}"
    reward_type_per_gpu="${reward_type_per_gpu:-}"
    
    # 使用单引号包裹值，避免特殊字符问题（bash 的 source 命令支持单引号）
    # 注意：如果值中包含单引号，需要特殊处理，但端点URL通常不包含单引号
    cat > "$export_file" << EOF
# 图像编辑服务端点（逗号分隔）
export EDIT_SERVER_ENDPOINTS="${edit_endpoints}"

# 奖励服务端点（clip/self_reward，根据 REWARD_TYPE 自动选择）
export REWARD_SERVER_ENDPOINTS="${selected_reward_endpoints}"

# CLIP 奖励服务端点（单独配置，用于 mixed 模式）
export CLIP_REWARD_SERVER_ENDPOINTS="${reward_endpoints_clip}"

# R3-Gen self-reward 服务端点（单独配置，用于 self_reward 或 mixed 模式）
export SELF_REWARD_SERVER_ENDPOINTS="${reward_endpoints_self_reward}"

# SAM3 奖励服务端点
export SAM3_REWARD_SERVER_ENDPOINTS="${reward_endpoints_sam3}"

# 奖励服务类型（self_reward、clip、sam3 或 mixed）
export REWARD_TYPE="${reward_type}"

# 按GPU指定的奖励类型
export REWARD_TYPE_PER_GPU="${reward_type_per_gpu}"

EOF
}

# 获取配置（供VERL训练节点使用）
get_config() {
    echo "========================================="
    echo "服务配置信息"
    echo "========================================="
    
    # 生成环境变量导出脚本
    local export_file="${CONFIG_DIR}/service_endpoints.env"
    mkdir -p "${CONFIG_DIR}"
    
    # 获取当前节点的IP地址（仅在配置文件不存在时使用，用于生成默认端点）
    local current_ip=$(get_server_address)
    echo "[INFO] 当前节点IP地址: $current_ip"
    
    # 读取编辑服务端点
    local edit_endpoints=$(read_edit_server_endpoints)
    
    # 读取奖励服务端点（分类读取）
    read_reward_server_endpoints
    local reward_endpoints_clip="$REWARD_ENDPOINTS_CLIP"
    local reward_endpoints_self_reward="$REWARD_ENDPOINTS_SELF_REWARD"
    local reward_endpoints_sam3="$REWARD_ENDPOINTS_SAM3"
    
    # 如果配置文件不存在或为空，生成默认端点
    generate_default_endpoints "$current_ip" "edit_endpoints" "reward_endpoints_clip" "reward_endpoints_self_reward" "reward_endpoints_sam3"
    
    # 获取奖励类型（从环境变量读取，默认 self_reward）
    local reward_type
    reward_type="$(normalize_reward_type "${REWARD_TYPE:-self_reward}")"
    local reward_type_per_gpu="${REWARD_TYPE_PER_GPU:-}"
    
    # 根据 REWARD_TYPE 选择正确的端点
    local selected_reward_endpoints=$(select_reward_endpoints \
        "$reward_type" \
        "$reward_type_per_gpu" \
        "$reward_endpoints_clip" \
        "$reward_endpoints_self_reward" \
        "$reward_endpoints_sam3")
    
    # 使用更新后的 reward_type（如果 select_reward_endpoints 检测到 SAM3 端点，可能会更新为 mixed）
    if [[ -n "${UPDATED_REWARD_TYPE:-}" ]]; then
        reward_type="$UPDATED_REWARD_TYPE"
    fi
    
    # 生成 service_endpoints.env 文件
    generate_service_endpoints_env \
        "$export_file" \
        "$edit_endpoints" \
        "$selected_reward_endpoints" \
        "$reward_endpoints_clip" \
        "$reward_endpoints_self_reward" \
        "$reward_endpoints_sam3" \
        "$reward_type" \
        "$reward_type_per_gpu"
    
    echo "配置文件已生成: $export_file"
    echo ""
    echo "在VERL训练节点上，执行以下命令导入配置："
    echo "  source $export_file"
    echo ""
    cat "$export_file"
}

# 显示帮助
show_help() {
    cat << EOF
多节点分布式服务部署脚本

用法:
  $0 <command> [options]

命令:
  edit_server           - 启动图像编辑服务（BAGEL 或 Qwen-Image-Edit）
                          可选参数: <model_path> <edit_model_type>
                          示例: $0 edit_server /path/to/model bagel
                          示例: $0 edit_server /path/to/model qwen_image_edit
  
  reward_server         - 启动奖励服务（支持self_reward/CLIP/SAM3/mixed）
                          self_reward 的具体模型由 SELF_REWARD_MODEL_TYPE 指定
                          支持: omniverifier, qwen2_5vl, qwen3vl
                          示例: SELF_REWARD_MODEL_TYPE=qwen3vl $0 reward_server
                          示例: REWARD_TYPE=mixed REWARD_TYPE_PER_GPU="0:self_reward,1:self_reward,2:clip,3:sam3" $0 reward_server

  stop                  - 停止所有服务
  stop_edit             - 只停止图像编辑服务
  stop_reward           - 只停止奖励服务
  
  get_config            - 生成服务配置文件（在任何节点运行）
  
  help                  - 显示此帮助信息

架构说明:
  节点0:   VERL训练（单节点模式，8 GPU）
  节点1-N: 图像编辑服务（BAGEL 或 Qwen-Image-Edit）
  节点M:   奖励服务（self_reward/CLIP/SAM3/mixed，根据REWARD_TYPE和REWARD_TYPE_PER_GPU指定）

环境变量:
  EDIT_MODEL_PATH       - 编辑模型路径（从 config.yaml 的 model.edit_model_path 加载）
  EDIT_MODEL_TYPE       - 图像编辑模型类型：bagel 或 qwen_image_edit
  SELF_REWARD_MODEL_PATH - self_reward模型路径
  SELF_REWARD_MODEL_TYPE - self_reward模型类型：omniverifier、qwen2_5vl 或 qwen3vl
  CLIP_MODULE_PATH      - 自定义CLIP模块路径（可选，不设置则使用标准clip包）
  CLIP_MODEL_PATH       - CLIP模型checkpoint路径（可选，.pt文件，用于加载自定义训练的模型）
  GPUS_PER_NODE         - 每个节点使用的最大GPU数量（可选，0表示使用所有检测到的GPU）
  EDIT_SERVER_NODE      - 图像编辑服务节点地址（默认: 当前节点hostname）
  REWARD_SERVER_NODE    - 奖励服务节点地址（默认: 当前节点hostname）

示例部署流程:
  # 在服务节点上启动图像编辑服务（单节点测试）
  EDIT_MODEL_TYPE=bagel ./deploy_services.sh edit_server
  
  # 在服务节点上启动奖励服务（单节点测试）
  REWARD_TYPE=self_reward SELF_REWARD_MODEL_TYPE=qwen3vl ./deploy_services.sh reward_server
  
  # 在多节点部署时，在训练节点上获取配置
  ./deploy_services.sh get_config
  
  # 在服务节点上导入配置
  source config/service_endpoints.env
  
  # 在训练节点上启动VERL训练
  ./train_quick.sh
EOF
}

# ========== 主逻辑 ==========
case "$SERVICE_TYPE" in
    edit_server)
        model_path="${2:-${EDIT_MODEL_PATH:-/path/to/BAGEL-7B-MoT}}"
        edit_model_type="${3:-${EDIT_MODEL_TYPE:-bagel}}"
        start_edit_server "$model_path" "$edit_model_type"

        echo "========================================="
        echo "输出信息重定向到log文件"
        echo "========================================="

        sleep $SERVICE_KEEPALIVE_SECONDS
        echo "代码退出"
        ;;
    reward_server)
        model_name="${2:-ViT-B/32}"  # 仅用于CLIP，第一个参数作为模型名称
        start_reward_server "$model_name"
    
        echo "========================================="
        echo "输出信息重定向到log文件"
        echo "========================================="
 
        sleep $SERVICE_KEEPALIVE_SECONDS
        echo "代码退出"
        ;;
    stop)
        stop_services
        ;;
    stop_edit)
        stop_edit_server
        ;;
    stop_reward)
        stop_reward_server
        ;;
    get_config)
        get_config
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "[ERROR] 未知命令: $SERVICE_TYPE"
        echo ""
        show_help
        exit 1
        ;;
esac
