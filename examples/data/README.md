# Data Format

Datasets are not included in this repository. Put local JSON/JSONL files and images here locally, or point the launch scripts to external paths with:

```bash
export TRAIN_DATA_PATH=/path/to/train.json
export VAL_DATA_PATH=/path/to/val.json
export IMAGE_DIR=/path/to/images
```

The default config uses `data.answer_key=ground_truth`, so each item should contain at least:

```json
{
  "prompt": "Describe the target image or verification question.",
  "images": ["relative/or/absolute/image.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"object\", \"prompt\": \"...\"}"
}
```

`ground_truth` is a JSON-encoded string. `images` may contain absolute paths or paths relative to `IMAGE_DIR`.

## Fields

- `prompt`: the original generation prompt or verification target. It is formatted by `examples/format_prompt/*.jinja` before being sent to the policy.
- `images`: a list of input image paths. The current training setup uses one image per item.
- `ground_truth.answer`: boolean label used by the first-stage MLLM judgment reward.
- `ground_truth.category`: optional task category, such as `object`, `color`, `shape`, `texture`, `spatial`, `numeracy`, `non`, or `complex`.
- `ground_truth.prompt`: recommended duplicate of the original prompt. It is useful for standalone reward calls and backward compatibility.
- `ground_truth.reward_type`: optional override for the second-stage reward, for example `self_reward`, `clip`, `sam3`, or `mixed`.

For direct MLLM judgment, `answer` is the only required field inside `ground_truth`. In practice, keep `category` and `prompt` too.

## Direct MLLM Reward

```json
{
  "prompt": "a red apple and a green orange",
  "images": ["val/color/02541/false/00002.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"color\", \"prompt\": \"a red apple and a green orange\"}"
}
```

## Question Decomposition Reward

The self-reward server also accepts decomposed yes/no questions. Store them in `ground_truth.generated_qa`:

```json
{
  "prompt": "a red apple and a green orange",
  "images": ["val/color/02541/false/00002.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"color\", \"prompt\": \"a red apple and a green orange\", \"reward_type\": \"self_reward\", \"generated_qa\": {\"yn_question_list\": [\"Is there a red apple in the image?\", \"Is there a green orange in the image?\"]}}"
}
```

If `generated_qa` is provided, the reward service scores the edited image by the fraction of yes/no questions answered true. This is mainly useful for MLLM self-reward backends, not CLIP.

## CLIP Reward

CLIP only needs the image and prompt. You can choose it globally with `REWARD_KWARGS`/`default_reward_type`, or per item:

```json
{
  "prompt": "a black candle and a white holder",
  "images": ["val/color/02801/false/00006.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"color\", \"prompt\": \"a black candle and a white holder\", \"reward_type\": \"clip\"}"
}
```

## SAM3 Reward

SAM3 needs object metadata in `ground_truth`. For `object`/`non` use `nouns`:

```json
{
  "prompt": "a photo of a red car",
  "images": ["val/object/00001.png"],
  "ground_truth": "{\"answer\": true, \"category\": \"object\", \"prompt\": \"a photo of a red car\", \"reward_type\": \"sam3\", \"nouns\": [\"red car\"]}"
}
```

For `color`, `shape`, and `texture`, use `attr_nouns` for attribute-aware targets:

```json
{
  "prompt": "a purple airplane and a pink toaster",
  "images": ["val/color/02175/false/00001.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"color\", \"prompt\": \"a purple airplane and a pink toaster\", \"reward_type\": \"sam3\", \"nouns\": [\"airplane\", \"toaster\"], \"attr_nouns\": [\"purple airplane\", \"pink toaster\"]}"
}
```

For `spatial`, add `spatial_info`:

```json
{
  "prompt": "the cup is on the left of the plate",
  "images": ["val/spatial/00001.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"spatial\", \"prompt\": \"the cup is on the left of the plate\", \"reward_type\": \"sam3\", \"nouns\": [\"cup\", \"plate\"], \"spatial_info\": {\"obj1\": \"cup\", \"obj2\": \"plate\", \"locality\": \"on the left of\"}}"
}
```

For `numeracy`, add `numeracy_info`:

```json
{
  "prompt": "a photo of three donuts",
  "images": ["val/numeracy/01807/false/00000.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"numeracy\", \"prompt\": \"a photo of three donuts\", \"reward_type\": \"sam3\", \"nouns\": [\"donut\"], \"numeracy_info\": [{\"obj_name\": \"donut\", \"num\": 3}]}"
}
```

For `complex`, use `nouns` and optionally `spatial_info`.

## Backward Compatibility

Older data may include `ground_truth.image_path`. The reward function can still use it as a fallback, but new data should rely on the top-level `images` field plus `IMAGE_DIR`.
