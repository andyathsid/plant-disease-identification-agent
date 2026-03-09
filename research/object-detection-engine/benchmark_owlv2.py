import os
import time
import random
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from transformers import Owlv2Processor


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "plantdoc"
OWLV2_MODEL_PATH = ROOT_DIR / "models" / "owlv2" / "owlv2.onnx"
OWLV2_PROCESSOR_PATH = ROOT_DIR / "models" / "owlv2"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

PLANTDOC_LABELS = ["leaf"]


def find_images(root: Path, max_count: int = 1000):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(Path(dirpath) / name)
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    random.shuffle(paths)
    return paths[:max_count]


def load_random_image(image_paths):
    path = random.choice(image_paths)
    img = Image.open(str(path)).convert("RGB")
    return img, path


def create_session(model_path: Path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])


def compute_stats(latencies_ms):
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def benchmark_owlv2(model_path, processor_path, image_paths, labels, threshold=0.1, warmup=10, iters=100):
    print(f"\n=== Benchmarking OWLv2 ===")
    print(f"Model: {model_path}")
    print(f"Labels ({len(labels)}): {labels[:5]}...")

    session = create_session(model_path)
    processor = Owlv2Processor.from_pretrained(str(processor_path), use_fast=False, local_files_only=True)

    pre_times, infer_times, post_times, total_times = [], [], [], []

    for i in range(warmup + iters):
        image, _ = load_random_image(image_paths)

        t0 = time.perf_counter()
        inputs_pt = processor(text=[labels], images=image, return_tensors="pt")
        inputs_np = {
            "pixel_values": inputs_pt["pixel_values"].numpy(),
            "input_ids": inputs_pt["input_ids"].numpy(),
            "attention_mask": inputs_pt["attention_mask"].numpy(),
        }
        t1 = time.perf_counter()

        raw_outputs = session.run(None, inputs_np)
        t2 = time.perf_counter()

        class ONNXOutput:
            def __init__(self, logits, pred_boxes):
                self.logits = logits
                self.pred_boxes = pred_boxes

        outputs = ONNXOutput(
            logits=torch.from_numpy(raw_outputs[0]),
            pred_boxes=torch.from_numpy(raw_outputs[1]),
        )
        _ = processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=[image.size[::-1]],
            threshold=threshold,
        )
        t3 = time.perf_counter()

        if i >= warmup:
            pre_ms = (t1 - t0) * 1000.0
            infer_ms = (t2 - t1) * 1000.0
            post_ms = (t3 - t2) * 1000.0
            pre_times.append(pre_ms)
            infer_times.append(infer_ms)
            post_times.append(post_ms)
            total_times.append(pre_ms + infer_ms + post_ms)

        if (i + 1) % 10 == 0:
            phase = "warmup" if i < warmup else "bench"
            print(f"  Iter {i+1}/{warmup+iters} ({phase})", end="\r")

    print("\nDone.")

    results = {
        "pre": compute_stats(pre_times),
        "infer": compute_stats(infer_times),
        "post": compute_stats(post_times),
        "total": compute_stats(total_times),
    }

    print("Phase stats (ms):")
    for phase, stats in results.items():
        print(
            f"  {phase:6s} -> mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
            f"p90={stats['p90']:.3f}, p95={stats['p95']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}"
        )

    return results


def main():
    random.seed(420)
    np.random.seed(420)

    print(f"Using image root: {DATA_DIR}")
    image_paths = find_images(DATA_DIR)
    print(f"Found {len(image_paths)} images for sampling.")

    stats = benchmark_owlv2(
        model_path=OWLV2_MODEL_PATH,
        processor_path=OWLV2_PROCESSOR_PATH,
        image_paths=image_paths,
        labels=PLANTDOC_LABELS,
    )

    all_results = []
    for phase_key, phase_name in [
        ("pre", "T_pre"),
        ("infer", "T_infer"),
        ("post", "T_post"),
        ("total", "T_total"),
    ]:
        phase_stats = stats[phase_key]
        all_results.append({
            "Model": "OWLv2",
            "Input_Size": "768x768",
            "Phase": phase_name,
            "Mean": phase_stats["mean"],
            "Median": phase_stats["median"],
            "P90": phase_stats["p90"],
            "P95": phase_stats["p95"],
            "Min": phase_stats["min"],
            "Max": phase_stats["max"],
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = ROOT_DIR / f"owlv2_benchmark_results_{timestamp}.csv"

    fieldnames = ["Model", "Input_Size", "Phase", "Mean", "Median", "P90", "P95", "Min", "Max"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\nSaved results to {out_csv}")

    print("\nSummary (ms):")
    print(f"{'Model':10s} {'Input':9s} {'Phase':7s} {'Mean':>8s} {'P90':>8s} {'P95':>8s}")
    for row in all_results:
        print(
            f"{row['Model']:10s} {row['Input_Size']:9s} {row['Phase']:7s} "
            f"{row['Mean']:8.3f} {row['P90']:8.3f} {row['P95']:8.3f}"
        )


if __name__ == "__main__":
    main()
