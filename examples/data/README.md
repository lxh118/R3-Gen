# Data Format

Datasets are not included in this repository. Put local JSON files and images here or point the launch scripts to external paths with:

```bash
export TRAIN_DATA_PATH=/path/to/train.json
export VAL_DATA_PATH=/path/to/val.json
export IMAGE_DIR=/path/to/images
```

Each item should contain at least:

```json
{
  "prompt": "Describe the target image or verification question.",
  "images": ["relative/or/absolute/image.png"],
  "ground_truth": "{\"answer\": false, \"category\": \"object\", \"prompt\": \"...\"}"
}
```

`images` may contain absolute paths or paths relative to `IMAGE_DIR`. Reward functions may read extra fields from `ground_truth`, such as `category`, object names, spatial relation metadata, or expected counts.
