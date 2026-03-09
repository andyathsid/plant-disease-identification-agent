import os
import time
import random
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# -----------------------------
# Paths and configuration
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"D:\Workspace\Repository\thesis\research\object-detection-engine\data\plantdoc")

YOLOV11_DIR = Path(r"D:\Workspace\Repository\thesis\research\object-detection-engine\models\yolov11\simplified")  

# Update these patterns if your dataset is organised differently
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# -----------------------------
# Pre/post-processing (from onnx.ipynb)
# -----------------------------

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)


def preprocess_image_batched(img_bgr, input_size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_shape = img_rgb.shape
    img_resized, ratio, pad = letterbox(img_rgb, new_shape=(input_size, input_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch, original_shape, ratio, pad


def postprocess_detections_yolo(outputs, original_shape, ratio, pad, conf_threshold=0.25, iou_threshold=0.45):
    predictions = outputs[0][0]
    confident = predictions[predictions[:, 4] > conf_threshold]
    if len(confident) == 0:
        return []

    boxes = confident[:, :4].copy()
    confidences = confident[:, 4].copy()
    class_ids = confident[:, 5].astype(int)

    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, [0, 2]] /= ratio[0]
    boxes[:, [1, 3]] /= ratio[1]

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold)

    final = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            final.append({
                "bbox": boxes[i],
                "confidence": float(confidences[i]),
                "class_id": int(class_ids[i]),
            })
    return final





# -----------------------------
# Utility helpers
# -----------------------------


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
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img, path


def create_session(model_path: Path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8  
    so.inter_op_num_threads = 1  # minimize contention
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), sess_options=so, providers=providers)


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


# -----------------------------
# Benchmark core
# -----------------------------


def benchmark_model(name, input_size, model_path, image_paths, warmup=30, iters=300):
    print(f"\n=== Benchmarking {name} ({input_size}x{input_size}) ===")
    print(f"Model: {model_path}")

    session = create_session(model_path)
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    pre_times, infer_times, post_times, total_times = [], [], [], []

    for i in range(warmup + iters):
        img_bgr, _ = load_random_image(image_paths)

        t0 = time.perf_counter()
        input_tensor, original_shape, ratio, pad = preprocess_image_batched(img_bgr, input_size)
        t1 = time.perf_counter()

        outputs = session.run(output_names, {input_name: input_tensor})
        t2 = time.perf_counter()

        _ = postprocess_detections_yolo(outputs, original_shape, ratio, pad)
        t3 = time.perf_counter()

        if i >= warmup:
            pre_ms = (t1 - t0) * 1000.0
            infer_ms = (t2 - t1) * 1000.0
            post_ms = (t3 - t2) * 1000.0
            pre_times.append(pre_ms)
            infer_times.append(infer_ms)
            post_times.append(post_ms)
            total_times.append(pre_ms + infer_ms + post_ms)

        if (i + 1) % 50 == 0:
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


# -----------------------------
# Main
# -----------------------------


def main():
    random.seed(420)
    np.random.seed(420)

    print(f"Using image root: {DATA_DIR}")
    image_paths = find_images(DATA_DIR)
    print(f"Found {len(image_paths)} images for sampling.")

    models = []

    # YOLOv11 ONNX (use 416x416 for all)
    for size_name in ["n", "s", "m"]:
        onnx_file = YOLOV11_DIR / f"yolo11{size_name}.onnx"
        if onnx_file.exists():
            models.append((f"YOLOv11-{size_name}", 416, onnx_file))
        else:
            print(f"Warning: missing {onnx_file}")

    if not models:
        raise RuntimeError("No ONNX models found to benchmark.")

    all_results = []

    for model_name, input_size, model_path in models:
        stats = benchmark_model(model_name, input_size, model_path, image_paths)
        for phase_key, phase_name in [
            ("pre", "T_pre"),
            ("infer", "T_infer"),
            ("post", "T_post"),
            ("total", "T_total"),
        ]:
            phase_stats = stats[phase_key]
            all_results.append({
                "Model": model_name,
                "Input_Size": f"{input_size}x{input_size}",
                "Phase": phase_name,
                "Mean": phase_stats["mean"],
                "Median": phase_stats["median"],
                "P90": phase_stats["p90"],
                "P95": phase_stats["p95"],
                "Min": phase_stats["min"],
                "Max": phase_stats["max"],
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = ROOT_DIR / f"onnx_benchmark_results_{timestamp}.csv"

    fieldnames = ["Model", "Input_Size", "Phase", "Mean", "Median", "P90", "P95", "Min", "Max"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\nSaved results to {out_csv}")

    print("\nSummary (ms):")
    print(f"{'Model':25s} {'Input':9s} {'Phase':7s} {'Mean':>8s} {'P90':>8s} {'P95':>8s}")
    for row in all_results:
        print(
            f"{row['Model']:25s} {row['Input_Size']:9s} {row['Phase']:7s} "
            f"{row['Mean']:8.3f} {row['P90']:8.3f} {row['P95']:8.3f}"
        )


if __name__ == "__main__":
    main()
