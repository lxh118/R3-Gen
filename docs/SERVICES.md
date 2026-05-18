# Distributed Services

R3-Gen keeps heavyweight image-edit and reward models in long-running HTTP services. Training workers call these services through `distributed_services/clients/api_client.py`, so the training environment does not need to load the edit model or the reward model weights.

## Service Layout

A typical run uses three roles:

1. Training node: runs VERL and sources `distributed_services/config/service_endpoints.env`.
2. Image-edit node: runs BAGEL or Qwen-Image-Edit services.
3. Reward node: runs self-reward, CLIP, SAM3, or a mixed set of reward services.

The roles can share one machine for debugging. For full training, run the edit and reward services on machines with the required model weights and runtime dependencies.

## Runtime Files

`deploy_services.sh` writes local endpoint files under `distributed_services/config/`:

```text
edit_server_endpoints.txt
reward_server_endpoints.txt
clip_reward_server_endpoints.txt
self_reward_server_endpoints.txt
sam3_reward_server_endpoints.txt
service_endpoints.env
```

The `*_endpoints.txt` files contain machine-specific IPs and ports. They are generated runtime artifacts and are ignored by git.

`service_endpoints.env` is also generated locally and ignored by git. `service_endpoints.example.env` is the committed template.

Generate the real file with:

```bash
bash distributed_services/scripts/deploy_services.sh get_config
source distributed_services/config/service_endpoints.env
```

`train_quick.sh` also sources `service_endpoints.env` automatically if the file exists.

If your service nodes and training node do not share the same workspace, copy the endpoint values manually or export the endpoint variables directly on the training node.

## Endpoint Variables

The training process reads these variables:

```bash
export EDIT_SERVER_ENDPOINTS="http://edit-node:5001,http://edit-node:5003"
export REWARD_SERVER_ENDPOINTS="http://reward-node:6001,http://reward-node:6002"
export CLIP_REWARD_SERVER_ENDPOINTS=""
export SELF_REWARD_SERVER_ENDPOINTS="http://reward-node:6001"
export SAM3_REWARD_SERVER_ENDPOINTS=""
export REWARD_TYPE="self_reward"
export REWARD_TYPE_PER_GPU=""
```

`REWARD_SERVER_ENDPOINTS` is the selected reward endpoint list for the current `REWARD_TYPE`. The type-specific variables are kept so mixed reward setups can route requests by reward type.

## Image-Edit Services

Set `EDIT_MODEL_TYPE` to choose the backend:

```bash
# BAGEL
export EDIT_MODEL_PATH=/path/to/BAGEL-7B-MoT
export EDIT_MODEL_TYPE=bagel
bash distributed_services/scripts/deploy_services.sh edit_server "$EDIT_MODEL_PATH" "$EDIT_MODEL_TYPE"
```

```bash
# Qwen-Image-Edit
export EDIT_MODEL_PATH=/path/to/Qwen-Image-Edit
export EDIT_MODEL_TYPE=qwen_image_edit
bash distributed_services/scripts/deploy_services.sh edit_server "$EDIT_MODEL_PATH" "$EDIT_MODEL_TYPE"
```

The image-edit client calls `/edit` and expects a base64 PNG response. Each server also exposes `/health`.

## Reward Services

Use `REWARD_TYPE=self_reward` for the R3-Gen paper method. The concrete self-reward backend is selected separately by `SELF_REWARD_MODEL_TYPE`.

```bash
export REWARD_TYPE=self_reward
export SELF_REWARD_MODEL_PATH=/path/to/self_reward_model
export SELF_REWARD_MODEL_TYPE=qwen3vl  # omniverifier, qwen2_5vl, or qwen3vl
bash distributed_services/scripts/deploy_services.sh reward_server
```

Do not set `REWARD_TYPE` to the backend name. For example, `qwen3vl` is a self-reward model type, not a reward type.

CLIP reward:

```bash
export REWARD_TYPE=clip
bash distributed_services/scripts/deploy_services.sh reward_server ViT-B/32
```

SAM3 reward:

```bash
export REWARD_TYPE=sam3
export SAM3_BPE_PATH=/path/to/bpe_simple_vocab_16e6.txt.gz
export SAM3_CKPT_PATH=/path/to/sam3.pt
export SAM3_METADATA_JSONL=/path/to/metadata.jsonl
bash distributed_services/scripts/deploy_services.sh reward_server
```

Mixed reward assignment:

```bash
export REWARD_TYPE=mixed
export REWARD_TYPE_PER_GPU="0:self_reward,1:self_reward,2:clip,3:sam3"
bash distributed_services/scripts/deploy_services.sh reward_server
```

In mixed mode, `REWARD_TYPE_PER_GPU` maps local GPU ids to reward types. The reward function uses this mapping to pick the correct service client for each worker.

## Training Node

After the services are running:

```bash
bash distributed_services/scripts/deploy_services.sh get_config
source distributed_services/config/service_endpoints.env

export MODEL_PATH=/path/to/policy_or_base_vlm
export TRAIN_DATA_PATH=/path/to/train.json
export VAL_DATA_PATH=/path/to/val.json
export IMAGE_DIR=/path/to/images

bash distributed_services/scripts/train_quick.sh
```

The default training script uses:

```bash
export FORMAT_PROMPT=examples/format_prompt/qwen3_vl_edit_optimized.jinja
export REWARD_FUNCTION=examples/reward_function/self_reward_staged_reward_api.py:compute_score
```

For LLaVA-OneVision training, use:

```bash
bash distributed_services/scripts/train_llava_onevision.sh
```

## Checks

Check service health:

```bash
curl http://edit-node:5001/health
curl http://reward-node:6001/health
```

Inspect generated endpoints:

```bash
bash distributed_services/scripts/deploy_services.sh get_config
cat distributed_services/config/service_endpoints.env
```

Stop services:

```bash
bash distributed_services/scripts/deploy_services.sh stop_edit
bash distributed_services/scripts/deploy_services.sh stop_reward
bash distributed_services/scripts/deploy_services.sh stop
```

Logs are written under `distributed_services/logs/`.
