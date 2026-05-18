# Distributed Services

R3-Gen runs image editing and reward scoring as long-running HTTP services. The training process only talks to these services through endpoint environment variables, so the training node does not need to load the edit model or reward model weights.

The deployment flow is simple:

1. Start the image-edit service on one or more edit nodes.
2. Start the reward service on one or more reward nodes.
3. On the training node, generate/source the endpoint environment file.
4. Start training.

The edit node, reward node, and training node can be the same machine for debugging. For full runs, they are usually separate GPU machines.

## 1. Start Image-Edit Service

Run this on the edit-service machine:

```bash
cd /path/to/R3-Gen

export EDIT_MODEL_PATH=/path/to/BAGEL-7B-MoT
export EDIT_MODEL_TYPE=bagel  # bagel or qwen_image_edit

bash distributed_services/scripts/deploy_services.sh edit_server \
  "$EDIT_MODEL_PATH" "$EDIT_MODEL_TYPE"
```

For Qwen-Image-Edit, use:

```bash
export EDIT_MODEL_PATH=/path/to/Qwen-Image-Edit
export EDIT_MODEL_TYPE=qwen_image_edit
```

The edit service exposes `/edit` and `/health`.

Shortcut wrappers are also available:

```bash
export EDIT_MODEL_PATH=/path/to/BAGEL-7B-MoT
bash distributed_services/scripts/start_bagel_simple.sh

export EDIT_MODEL_PATH=/path/to/Qwen-Image-Edit
bash distributed_services/scripts/start_qwen_simple.sh
```

## 2. Start Reward Service

Run this on the reward-service machine:

```bash
cd /path/to/R3-Gen

export REWARD_TYPE=self_reward
export SELF_REWARD_MODEL_PATH=/path/to/self_reward_model
export SELF_REWARD_MODEL_TYPE=qwen3vl  # omniverifier, qwen2_5vl, or qwen3vl

bash distributed_services/scripts/deploy_services.sh reward_server
```

For the R3-Gen paper method, keep `REWARD_TYPE=self_reward`. `SELF_REWARD_MODEL_TYPE` only selects which model backend serves the self-reward.

Other reward backends use the same command with a different `REWARD_TYPE`:

```bash
# CLIP reward
export REWARD_TYPE=clip
bash distributed_services/scripts/deploy_services.sh reward_server ViT-B/32

# SAM3 reward
export REWARD_TYPE=sam3
export SAM3_BPE_PATH=/path/to/bpe_simple_vocab_16e6.txt.gz
export SAM3_CKPT_PATH=/path/to/sam3.pt
bash distributed_services/scripts/deploy_services.sh reward_server
```

The reward service exposes `/reward` and `/health`.

## 3. Prepare Training Endpoints

When a service starts, `deploy_services.sh` records its URL under `distributed_services/config/`. On the training node, turn those local endpoint records into the environment file used by training:

```bash
cd /path/to/R3-Gen

bash distributed_services/scripts/deploy_services.sh get_config
source distributed_services/config/service_endpoints.env
```

If the service nodes and training node share the same filesystem, this is enough.

If they do not share a filesystem, copy the generated endpoint text files from the edit/reward nodes into the training node's `distributed_services/config/` directory before running `get_config`, or manually export the endpoints:

```bash
export EDIT_SERVER_ENDPOINTS="http://edit-node:5001,http://edit-node:5002"
export REWARD_TYPE=self_reward
export REWARD_SERVER_ENDPOINTS="http://reward-node:6001,http://reward-node:6002"
export SELF_REWARD_SERVER_ENDPOINTS="$REWARD_SERVER_ENDPOINTS"
```

`get_config` only collects endpoints that already exist in endpoint files. It does not guess service addresses from the training node IP.

`service_endpoints.env` and the `*_endpoints.txt` files are machine-local runtime files and are ignored by git.

## 4. Start Training

Run this on the training machine after the services are reachable:

```bash
cd /path/to/R3-Gen
source distributed_services/config/service_endpoints.env

export MODEL_PATH=/path/to/policy_or_base_vlm
export TRAIN_DATA_PATH=/path/to/train.json
export VAL_DATA_PATH=/path/to/val.json
export IMAGE_DIR=/path/to/images

bash distributed_services/scripts/train_quick.sh
```

The default training entry uses:

```bash
export FORMAT_PROMPT=examples/format_prompt/qwen3_vl_edit_optimized.jinja
export REWARD_FUNCTION=examples/reward_function/self_reward_staged_reward_api.py:compute_score
```

For LLaVA-OneVision training:

```bash
bash distributed_services/scripts/train_llava_onevision.sh
```

## Scaling

To use more GPUs on one service node, set `GPUS_PER_NODE` or edit `distributed_services/config/config.yaml` before starting the service. To use more service nodes, run the same edit or reward startup command on additional machines and collect their endpoint files on the training node before `get_config`.

For mixed reward serving, map local GPU ids to reward types:

```bash
export REWARD_TYPE=mixed
export REWARD_TYPE_PER_GPU="0:self_reward,1:self_reward,2:clip,3:sam3"
bash distributed_services/scripts/deploy_services.sh reward_server
```

Training can also use a multi-node Ray cluster. Start Ray on the training head and worker nodes first, then run training from the head node with `NNODES` and `N_GPUS_PER_NODE` set:

```bash
# Head training node
ray stop --force
ray start --head

# Worker training node
ray stop --force
ray start --address='HEAD_NODE_IP:6379'
bash distributed_services/scripts/ray.sh  # optional keep-alive helper

# Back on the head node
export NNODES=2
export N_GPUS_PER_NODE=4
source distributed_services/config/service_endpoints.env
bash distributed_services/scripts/train_quick.sh
```

`ray.sh` does not start Ray; it only keeps a worker-node shell or scheduler job alive after `ray start --address=...`.

## Checks

```bash
curl http://edit-node:5001/health
curl http://reward-node:6001/health

bash distributed_services/scripts/deploy_services.sh stop_edit
bash distributed_services/scripts/deploy_services.sh stop_reward
```

Service logs are written under `distributed_services/logs/`.
