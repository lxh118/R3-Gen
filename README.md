# Overview

Official code release for **Benchmarking and Evolving
Reason-Reflect-Rectify for Reflective Visual Generation**.

<p align="center">
  <img src="assets/r3_refiner_pipeline.png" alt="R3-Refiner training pipeline" width="95%">
</p>

R3-Gen trains a reflective vision-language policy with online image-edit rollouts and reward feedback.

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
pip install -e .
```

Some optional services require extra model-specific dependencies:

- BAGEL edit server: BAGEL runtime and model weights.
- Qwen image-edit server: `diffusers` with Qwen-Image-Edit support and optional `cache-dit`.
- SAM3 reward server: SAM3 package and checkpoints.

## Prepare Data

Place your training JSON files and images outside the repository or under `examples/data/` locally. See [examples/data/README.md](examples/data/README.md) for the expected schema.

Typical variables:

```bash
export TRAIN_DATA_PATH=/path/to/train.json
export VAL_DATA_PATH=/path/to/val.json
export IMAGE_DIR=/path/to/images
```

## Start Services

Edit `distributed_services/config/config.yaml` or export equivalent environment variables for local model paths.

Start an image-edit service on an edit node:

```bash
export EDIT_MODEL_PATH=/path/to/BAGEL-7B-MoT
bash distributed_services/scripts/deploy_services.sh edit_server "$EDIT_MODEL_PATH" bagel
```

Start an OmniVerifier/R3-Gen reward service on a reward node:

```bash
export OMNIVERIFIER_MODEL_PATH=/path/to/reward_model
export REWARD_TYPE=omniverifier
bash distributed_services/scripts/deploy_services.sh reward_server
```

Generate endpoint environment variables for the training node:

```bash
bash distributed_services/scripts/deploy_services.sh get_config
source distributed_services/config/service_endpoints.env
```

More details are in [docs/SERVICES.md](docs/SERVICES.md).

## Train

Run the default VLM training entry:

```bash
export MODEL_PATH=/path/to/policy_or_base_vlm
export TRAIN_DATA_PATH=/path/to/train.json
export VAL_DATA_PATH=/path/to/val.json
export IMAGE_DIR=/path/to/images

bash distributed_services/scripts/train_quick.sh
```

Common overrides:

```bash
export PROJECT_NAME=R3-Gen
export EXPERIMENT_NAME=r3-gen-grpo
export N_GPUS_PER_NODE=8
export ROLLOUT_BATCH_SIZE=128
export FORMAT_PROMPT=examples/format_prompt/qwen3_vl_edit_optimized.jinja
export REWARD_FUNCTION=examples/reward_function/qwen3vl_staged_reward_api.py:compute_score
```

For a LLaVA-OneVision policy, use:

```bash
bash distributed_services/scripts/train_llava_onevision.sh
```

## R3-Bench

<p align="center">
  <img src="assets/r3_bench_overview.png" alt="R3-Bench overview" width="95%">
</p>


## Qualitative Examples

<p align="center">
  <img src="assets/qualitative_comparison.png" alt="Qualitative comparison of R3-Refiner" width="95%">
</p>
