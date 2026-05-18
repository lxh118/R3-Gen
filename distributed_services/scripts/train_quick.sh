#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

export VLLM_DISABLE_SYMMETRIC_MEMORY="${VLLM_DISABLE_SYMMETRIC_MEMORY:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

project_name="${PROJECT_NAME:-R3-Gen}"
experiment_name="${EXPERIMENT_NAME:-r3-gen-grpo}"
model_path="${MODEL_PATH:?Set MODEL_PATH to the policy checkpoint or base VLM.}"

train_data_path="${TRAIN_DATA_PATH:-examples/data/demo_train.json}"
val_data_path="${VAL_DATA_PATH:-examples/data/demo_train.json}"
image_dir="${IMAGE_DIR:-examples/data/images}"
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-checkpoints/${project_name}/${experiment_name}}"

config_file="${CONFIG_FILE:-distributed_services/config/config.yaml}"
verl_base_config_rel="${VERL_BASE_CONFIG:-distributed_services/config/verl_config.yaml}"
format_prompt="${FORMAT_PROMPT:-examples/format_prompt/qwen3_vl_edit_optimized.jinja}"
reward_function="${REWARD_FUNCTION:-examples/reward_function/self_reward_staged_reward_api.py:compute_score}"

if [[ -n "${REWARD_KWARGS:-}" ]]; then
    reward_kwargs="${REWARD_KWARGS}"
else
    reward_kwargs="{\"think_format_weight\":0.1,\"json_format_weight\":0.05,\"stage1_weight\":0.2,\"stage2_weight\":0.8,\"image_dir\":\"${image_dir}\",\"default_reward_type\":\"self_reward\",\"enable_stage2\":true}"
fi

if [[ -n "${ROLLOUT_BATCH_SIZE:-}" ]]; then
    rollout_batch_size="${ROLLOUT_BATCH_SIZE}"
elif [[ "${train_data_path}" == "examples/data/demo_train.json" ]]; then
    rollout_batch_size=64
else
    rollout_batch_size=128
fi
n_gpus_per_node="${N_GPUS_PER_NODE:-8}"
trainer_nnodes="${NNODES:-1}"
max_prompt_length="${MAX_PROMPT_LENGTH:-4096}"
max_response_length="${MAX_RESPONSE_LENGTH:-2048}"
actor_micro_batch_update="${ACTOR_MICRO_BATCH_UPDATE:-4}"
actor_micro_batch_experience="${ACTOR_MICRO_BATCH_EXPERIENCE:-4}"

train_log_file="${LOG_FILE:-distributed_services/logs/${project_name}_${experiment_name}_$(date +"%Y%m%d_%H%M%S").log}"
mkdir -p "$(dirname "${train_log_file}")"

if [[ -f "distributed_services/config/service_endpoints.env" ]]; then
    source distributed_services/config/service_endpoints.env
fi

python_bin="${PYTHON_BIN:-python3}"
config_exports="$("${python_bin}" distributed_services/config/load_config.py --config "${config_file}")"
eval "${config_exports}"
if [[ "${HF_ENDPOINT+x}" == "x" && -z "${HF_ENDPOINT}" ]]; then
    unset HF_ENDPOINT
fi

"${python_bin}" -m verl.trainer.main \
    config="${verl_base_config_rel}" \
    data.image_dir="${image_dir}" \
    "data.train_files='${train_data_path}'" \
    "data.val_files='${val_data_path}'" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.rollout_batch_size="${rollout_batch_size}" \
    data.val_filter_overlong_prompts=false \
    data.train_filter_overlong_prompts=false \
    data.format_prompt="${format_prompt}" \
    worker.actor.model.model_path="${model_path}" \
    worker.actor.model.trust_remote_code=true \
    worker.actor.use_torch_compile=false \
    worker.actor.micro_batch_size_per_device_for_update="${actor_micro_batch_update}" \
    worker.actor.micro_batch_size_per_device_for_experience="${actor_micro_batch_experience}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.nnodes="${trainer_nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.save_checkpoint_path="${save_checkpoint_path}" \
    worker.reward.reward_function="${reward_function}" \
    worker.reward.reward_function_kwargs="${reward_kwargs}" \
    "$@" 2>&1 | tee "${train_log_file}"
