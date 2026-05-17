# Distributed Services

R3-Gen keeps heavy image-edit and reward models in long-running HTTP services. Training workers call them through `distributed_services/clients/api_client.py`.

## Endpoint Variables

The training process reads these variables directly or through `distributed_services/config/service_endpoints.env`:

```bash
export EDIT_SERVER_ENDPOINTS="http://edit-node:5001,http://edit-node:5003"
export REWARD_SERVER_ENDPOINTS="http://reward-node:6001,http://reward-node:6002"
export CLIP_REWARD_SERVER_ENDPOINTS=""
export OMNIVERIFIER_REWARD_SERVER_ENDPOINTS="http://reward-node:6001"
export SAM3_REWARD_SERVER_ENDPOINTS=""
export REWARD_TYPE="omniverifier"
export REWARD_TYPE_PER_GPU=""
```

`deploy_services.sh get_config` can build `service_endpoints.env` from endpoint text files under `distributed_services/config/`.

## Image-Edit Services

BAGEL:

```bash
export EDIT_MODEL_PATH=/path/to/BAGEL-7B-MoT
bash distributed_services/scripts/deploy_services.sh edit_server "$EDIT_MODEL_PATH" bagel
```

Qwen Image Edit:

```bash
export EDIT_MODEL_PATH=/path/to/Qwen-Image-Edit
bash distributed_services/scripts/deploy_services.sh edit_server "$EDIT_MODEL_PATH" qwen
```

The image-edit client calls `/edit` and expects a base64 PNG response.

## Reward Services

OmniVerifier/R3-Gen reward:

```bash
export OMNIVERIFIER_MODEL_PATH=/path/to/reward_model
export REWARD_TYPE=omniverifier
bash distributed_services/scripts/deploy_services.sh reward_server
```

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

Mixed reward assignment can be controlled with `REWARD_TYPE_PER_GPU`, for example:

```bash
export REWARD_TYPE=mixed
export REWARD_TYPE_PER_GPU="0:omniverifier,1:omniverifier,2:clip,3:sam3"
```

## Training Node

After the services are up:

```bash
bash distributed_services/scripts/deploy_services.sh get_config
source distributed_services/config/service_endpoints.env
bash distributed_services/scripts/train_quick.sh
```

Each server exposes `/health`. The client uses round-robin routing, retries, passive health tracking, and endpoint hot reload from `service_endpoints.env`.
