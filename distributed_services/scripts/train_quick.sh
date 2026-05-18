#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

export VLLM_DISABLE_SYMMETRIC_MEMORY="${VLLM_DISABLE_SYMMETRIC_MEMORY:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

project_name="${PROJECT_NAME:-R3-Gen}"
experiment_name="${EXPERIMENT_NAME:-r3-gen-grpo}"
model_path="${MODEL_PATH:?Set MODEL_PATH to the policy checkpoint or base VLM.}"

train_data_path="${TRAIN_DATA_PATH:-examples/data/train.json}"
val_data_path="${VAL_DATA_PATH:-examples/data/val.json}"
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

rollout_batch_size="${ROLLOUT_BATCH_SIZE:-128}"
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

if [[ "${RUN_PREFLIGHT_CHECKS:-1}" != "0" ]]; then
    PREFLIGHT_MODEL_PATH="${model_path}" \
    PREFLIGHT_TRAIN_FILES="${train_data_path}" \
    PREFLIGHT_VAL_FILES="${val_data_path}" \
    PREFLIGHT_IMAGE_DIR="${image_dir}" \
    PREFLIGHT_REWARD_KWARGS="${reward_kwargs}" \
    "${python_bin}" - <<'PY'
import json
import os
import sys


def fail(message: str) -> None:
    print(f"[preflight][ERROR] {message}", file=sys.stderr)
    sys.exit(1)


def split_specs(value: str) -> list[str]:
    specs = []
    for item in (value or "").split(","):
        item = item.strip().strip("'").strip('"')
        if not item:
            continue
        specs.append(item.split("@", 1)[0])
    return specs


def load_records(path: str, limit: int = 64) -> list[dict]:
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    if len(records) >= limit:
                        break
        return records

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[:limit]
    if isinstance(data, dict):
        for key in ("data", "train", "validation", "val"):
            value = data.get(key)
            if isinstance(value, list):
                return value[:limit]
    return []


model_path = os.environ["PREFLIGHT_MODEL_PATH"]
train_files = split_specs(os.environ.get("PREFLIGHT_TRAIN_FILES", ""))
val_files = split_specs(os.environ.get("PREFLIGHT_VAL_FILES", ""))
image_dir = os.environ.get("PREFLIGHT_IMAGE_DIR", "")

if not os.path.exists(model_path):
    fail(f"MODEL_PATH does not exist: {model_path}")

all_files = train_files + val_files
if not train_files:
    fail("TRAIN_DATA_PATH is empty.")
for path in all_files:
    if not os.path.exists(path):
        fail(f"Data file does not exist: {path}")

if image_dir and not os.path.isdir(image_dir):
    fail(f"IMAGE_DIR does not exist or is not a directory: {image_dir}")

for data_file in all_files:
    for record in load_records(data_file):
        images = record.get("images")
        if not isinstance(images, list) or not images or not isinstance(images[0], str):
            continue
        image_path = images[0]
        if os.path.isabs(image_path):
            resolved = image_path
        else:
            if not image_dir:
                fail(f"Relative image path requires IMAGE_DIR. data={data_file}, image={image_path}")
            resolved = os.path.join(image_dir, image_path)
        if not os.path.exists(resolved):
            fail(f"Image file does not exist: {resolved} (from {data_file})")
        break

reward_kwargs = {}
try:
    reward_kwargs = json.loads(os.environ.get("PREFLIGHT_REWARD_KWARGS", "{}"))
except json.JSONDecodeError:
    pass

enable_stage2 = bool(reward_kwargs.get("enable_stage2", True))
if enable_stage2 and not os.environ.get("EDIT_SERVER_ENDPOINTS", "").strip():
    fail("EDIT_SERVER_ENDPOINTS is empty but stage-2 image editing is enabled.")

reward_type = os.environ.get("REWARD_TYPE", "self_reward").strip().lower().replace("-", "_")
reward_endpoints = os.environ.get("REWARD_SERVER_ENDPOINTS", "").strip()
self_reward_endpoints = os.environ.get("SELF_REWARD_SERVER_ENDPOINTS", "").strip()
if reward_type in {"self_reward", "mixed"} and not (reward_endpoints or self_reward_endpoints):
    fail("Self-reward endpoints are empty. Run reward_server/get_config or export SELF_REWARD_SERVER_ENDPOINTS.")

print(f"[preflight] OK: {len(train_files)} train file(s), {len(val_files)} val file(s).")
PY
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
