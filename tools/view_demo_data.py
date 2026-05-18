#!/usr/bin/env python3
"""Gradio viewer for the R3-Gen demo training data."""

import argparse
import json
import os
from pathlib import Path

# Some cluster shells export SOCKS proxies without httpx socks support. Gradio
# imports httpx at module import time, so clear them before importing gradio.
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(key, None)

import gradio as gr


def parse_args():
    parser = argparse.ArgumentParser(description="View R3-Gen demo training cases.")
    parser.add_argument("--data", default="examples/data/demo_train.json", help="Path to the demo JSON file.")
    parser.add_argument("--image-dir", default="examples/data/images", help="Root directory for relative image paths.")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio server host.")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    return parser.parse_args()


def load_records(data_path: Path, image_dir: Path):
    with data_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for idx, item in enumerate(raw):
        gt_raw = item.get("ground_truth", {})
        if isinstance(gt_raw, str):
            try:
                gt = json.loads(gt_raw)
            except json.JSONDecodeError:
                gt = {"raw_ground_truth": gt_raw}
        elif isinstance(gt_raw, dict):
            gt = gt_raw
        else:
            gt = {}

        images = item.get("images") or []
        image_value = images[0] if images else ""
        image_path = Path(image_value)
        if image_value and not image_path.is_absolute():
            image_path = image_dir / image_path

        records.append(
            {
                "idx": idx,
                "prompt": item.get("prompt") or gt.get("prompt", ""),
                "image": str(image_path) if image_value else None,
                "image_value": image_value,
                "ground_truth": gt,
                "category": gt.get("category", "unknown"),
                "answer": bool(gt.get("answer")) if isinstance(gt.get("answer"), bool) else gt.get("answer"),
                "data_source": gt.get("data_source", "unknown"),
                "construction_strategy": gt.get("construction_strategy", "unknown"),
            }
        )
    return records


def label(record):
    answer = str(record["answer"]).lower()
    prompt = record["prompt"].replace("\n", " ")
    if len(prompt) > 90:
        prompt = prompt[:87] + "..."
    return f"{record['idx']:03d} | {record['category']} | {answer} | {record['data_source']} | {prompt}"


def build_app(records):
    labels = {label(record): record["idx"] for record in records}
    by_idx = {record["idx"]: record for record in records}
    categories = ["all"] + sorted({record["category"] for record in records})
    sources = ["all"] + sorted({record["data_source"] for record in records})

    def filtered_labels(category, answer, source):
        values = []
        for record in records:
            if category != "all" and record["category"] != category:
                continue
            if answer != "all" and str(record["answer"]).lower() != answer:
                continue
            if source != "all" and record["data_source"] != source:
                continue
            values.append(label(record))
        return values

    def summary_text(category="all", answer="all", source="all"):
        values = filtered_labels(category, answer, source)
        return f"Showing {len(values)} / {len(records)} cases."

    def refresh(category, answer, source):
        values = filtered_labels(category, answer, source)
        value = values[0] if values else None
        return gr.update(choices=values, value=value), summary_text(category, answer, source)

    def show(selected):
        if not selected:
            return None, "", "", {}, ""
        record = by_idx[labels[selected]]
        exists = record["image"] and Path(record["image"]).exists()
        meta = (
            f"**Index:** {record['idx']}  \n"
            f"**Category:** {record['category']}  \n"
            f"**Answer:** {record['answer']}  \n"
            f"**Source:** {record['data_source']}  \n"
            f"**Construction:** {record['construction_strategy']}  \n"
            f"**Image:** `{record['image_value']}`  \n"
            f"**Image exists:** {exists}"
        )
        return record["image"] if exists else None, record["prompt"], meta, record["ground_truth"], record["image"] or ""

    with gr.Blocks(title="R3-Gen Demo Training Data") as demo:
        gr.Markdown("# R3-Gen Demo Training Data")
        gr.Markdown("Filter by dimension, label, and source to inspect the sampled training cases.")
        with gr.Row():
            category = gr.Dropdown(categories, value="all", label="Category")
            answer = gr.Radio(["all", "true", "false"], value="all", label="Answer")
            source = gr.Dropdown(sources, value="all", label="Data source")
        summary = gr.Markdown(summary_text())
        sample = gr.Dropdown(list(labels), value=list(labels)[0] if labels else None, label="Case")
        with gr.Row():
            image = gr.Image(type="filepath", label="Image", height=520)
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=5)
                meta = gr.Markdown()
                image_path = gr.Textbox(label="Resolved image path", interactive=False)
        gt = gr.JSON(label="ground_truth")

        for control in (category, answer, source):
            control.change(refresh, inputs=[category, answer, source], outputs=[sample, summary]).then(show, inputs=sample, outputs=[image, prompt, meta, gt, image_path])
        sample.change(show, inputs=sample, outputs=[image, prompt, meta, gt, image_path])
        demo.load(show, inputs=sample, outputs=[image, prompt, meta, gt, image_path])
    return demo


def main():
    args = parse_args()
    records = load_records(Path(args.data), Path(args.image_dir))
    app = build_app(records)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
