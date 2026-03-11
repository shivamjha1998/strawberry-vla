#!/usr/bin/env python3
"""
Train YOLO11 on Custom Strawberry Detection Dataset
====================================================
Optimized for Mac Mini M4 (16GB RAM) using MPS acceleration.
Uses YOLO11s for 2-class detection (ripe/unripe).
Dataset: Custom curated dataset (4536 images) in datasets/strawberry/

Usage:
    # Verify dataset is ready
    python train_strawberry_yolo.py --verify

    # Train
    python train_strawberry_yolo.py --train

    # Train with Small model (default)
    python train_strawberry_yolo.py --train --model-size s

    # Train with Nano model (faster, less accurate)
    python train_strawberry_yolo.py --train --model-size n

    # Resume interrupted training
    python train_strawberry_yolo.py --train --resume

    # Validate & test
    python train_strawberry_yolo.py --test

    # Run inference on a single image
    python train_strawberry_yolo.py --infer path/to/image.jpg

    # Show training summary
    python train_strawberry_yolo.py --summary

    # Full pipeline: verify → train → test
    python train_strawberry_yolo.py --verify --train --test
"""

import argparse
import os
import sys
import shutil
import time
from pathlib import Path
from collections import Counter

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "datasets" / "strawberry"
RUNS_DIR = PROJECT_DIR / "runs"

# Expected classes after relabeling
EXPECTED_CLASSES = ["ripe", "unripe"]

# ─── Training Config ─────────────────────────────────────────────────────────
# Optimized for:
#   - Mac Mini M4 16GB (MPS) or CUDA GPU
#   - 4536 images, 2 classes (ripe/unripe)
#   - YOLO11s (Small) — good balance of accuracy & speed for this dataset size

TRAIN_CONFIG = {
    # Model
    "model": "yolo11s.pt",           # YOLO11 Small — better accuracy than Nano
                                      # 4536 images is enough to benefit from 11s
    # Training
    "epochs": 100,                    # More data → can train longer
    "batch": 16,                      # Safe for 16GB RAM with Small model
    "imgsz": 640,                     # Standard YOLO input size
    "device": None,                   # Auto-detect (MPS/CUDA/CPU)
    "workers": 4,                     # M4 has efficient cores
    "patience": 25,                   # Early stopping — generous with larger dataset
    "save_period": 10,                # Save checkpoint every 10 epochs

    # Output
    "project": str(RUNS_DIR / "detect"),
    "name": "strawberry_v2",          # New name to avoid overwriting Phase 1 results
    "exist_ok": True,

    # Transfer learning
    "pretrained": True,               # Start from COCO pre-trained weights
    "optimizer": "AdamW",             # Best optimizer for fine-tuning
    "lr0": 0.001,                     # Lower LR for fine-tuning (not training from scratch)
    "lrf": 0.01,                      # Final LR = lr0 * lrf
    "cos_lr": True,                   # Cosine annealing schedule

    # ─── Augmentation (moderate, not heavy) ───
    # Key principle: ripeness depends on COLOR, so we keep color augmentations
    # moderate to avoid confusing ripe ↔ unripe during training.

    "cls": 1.5,                       # Class balance
    "mosaic": 1.0,                    # Mosaic: combines 4 images into 1
                                      # Great for learning spatial context
    "close_mosaic": 15,               # Turn off mosaic for last 15 epochs
                                      # Lets model fine-tune on clean images

    "hsv_h": 0.015,                   # Hue shift — very subtle
                                      # Too much would make red look orange/pink
    "hsv_s": 0.5,                     # Saturation — moderate
                                      # Simulates lighting variation without
                                      # destroying color-based ripeness cues
    "hsv_v": 0.3,                     # Brightness — moderate
                                      # Simulates greenhouse shadow/sunlight

    "flipud": 0.3,                    # Vertical flip — strawberries hang down
    "fliplr": 0.5,                    # Horizontal flip — standard
    "degrees": 15,                    # Rotation — slight tilt variation
    "scale": 0.4,                     # Scale — zoom in/out
    "translate": 0.1,                 # Translation — slight position shift

    "mixup": 0.1,                     # Blend two images — good regularization
    "copy_paste": 0.1,               # Copy-paste objects between images
                                      # Simulates more strawberries in scene

    "verbose": True,
}


# ─── Utility Functions ───────────────────────────────────────────────────────

def detect_device():
    """Auto-detect the best available compute device."""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon GPU) detected")
            return "mps"
        elif torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU detected: {gpu_name}")
            return "cuda"
        else:
            print("⚠️  No GPU found, using CPU (training will be slow)")
            return "cpu"
    except Exception:
        print("⚠️  Could not detect GPU, defaulting to CPU")
        return "cpu"


def find_data_yaml():
    """Find and return the path to data.yaml."""
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"❌ data.yaml not found at {data_yaml}")
        print("   Make sure your dataset is in datasets/strawberry/")
        sys.exit(1)
    return data_yaml


def fix_data_yaml():
    """
    Fix data.yaml paths and validate configuration.
    - Converts relative paths to absolute
    - Normalizes 'valid' → 'val'
    - Validates class names
    """
    import yaml

    data_yaml = find_data_yaml()

    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    modified = False

    # Fix paths to be absolute
    for key in ["train", "val", "valid", "test"]:
        if key in config:
            path = config[key]
            if not os.path.isabs(path):
                abs_path = str(DATASET_DIR / path)
                config[key] = abs_path
                modified = True

    # Normalize: Roboflow uses "valid" but YOLO expects "val"
    if "valid" in config and "val" not in config:
        config["val"] = config.pop("valid")
        modified = True

    if modified:
        with open(data_yaml, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print(f"✅ data.yaml ready: {data_yaml}")
    print(f"   Classes ({config.get('nc', '?')}): {config.get('names', 'unknown')}")

    return str(data_yaml)


def count_dataset_stats():
    """Count images and label distribution across splits."""
    import yaml

    data_yaml = find_data_yaml()
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    class_names = config.get("names", [])
    total_images = 0
    total_labels = 0
    class_counts = Counter()
    null_count = 0

    print(f"\n{'─'*60}")
    print(f"Dataset Statistics")
    print(f"{'─'*60}")
    print(f"  Location:  {DATASET_DIR}")
    print(f"  Classes:   {class_names}")
    print(f"  NC:        {config.get('nc', '?')}")

    for split in ["train", "valid", "val", "test"]:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        if not images_dir.exists():
            continue

        img_files = list(images_dir.glob("*.[jJpP][pPnN][gG]*"))
        split_images = len(img_files)
        split_labels = 0
        split_null = 0
        split_class_counts = Counter()

        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                content = label_file.read_text().strip()
                if not content:
                    # Empty label file = null/background image
                    split_null += 1
                else:
                    for line in content.split("\n"):
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                            split_class_counts[cls_name] += 1
                            split_labels += 1

        total_images += split_images
        total_labels += split_labels
        null_count += split_null
        class_counts.update(split_class_counts)

        print(f"\n  {split}:")
        print(f"    Images:     {split_images}")
        print(f"    Labels:     {split_labels}")
        print(f"    Null/BG:    {split_null}")
        for cls, count in sorted(split_class_counts.items()):
            print(f"    {cls}:  {count}")

    print(f"\n  {'─'*40}")
    print(f"  TOTAL:")
    print(f"    Images:     {total_images}")
    print(f"    Labels:     {total_labels}")
    print(f"    Null/BG:    {null_count}")
    for cls, count in sorted(class_counts.items()):
        pct = (count / total_labels * 100) if total_labels > 0 else 0
        print(f"    {cls}:  {count} ({pct:.1f}%)")

    # Warn about class imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        if max_count > 3 * min_count:
            print(f"\n  ⚠️  Class imbalance detected! Ratio: {max_count/min_count:.1f}:1")
            print(f"     Consider using class weights or targeted augmentation.")

    print(f"{'─'*60}\n")

    return {
        "total_images": total_images,
        "total_labels": total_labels,
        "null_count": null_count,
        "class_counts": dict(class_counts),
    }


# ─── Core Functions ──────────────────────────────────────────────────────────

def verify_dataset():
    """Verify dataset structure and print statistics."""
    print(f"\n{'='*60}")
    print(f"Verifying Dataset")
    print(f"{'='*60}")

    # Check directory structure
    required = ["train/images", "train/labels"]
    recommended = ["valid/images", "valid/labels", "test/images", "test/labels"]

    all_good = True
    for path in required:
        full_path = DATASET_DIR / path
        if full_path.exists():
            print(f"  ✅ {path}")
        else:
            # Also check 'val' variant
            alt_path = path.replace("valid", "val")
            alt_full = DATASET_DIR / alt_path
            if alt_full.exists():
                print(f"  ✅ {alt_path}")
            else:
                print(f"  ❌ {path} — MISSING (required)")
                all_good = False

    for path in recommended:
        full_path = DATASET_DIR / path
        if full_path.exists():
            print(f"  ✅ {path}")
        else:
            alt_path = path.replace("valid", "val")
            alt_full = DATASET_DIR / alt_path
            if alt_full.exists():
                print(f"  ✅ {alt_path}")
            else:
                print(f"  ⚠️  {path} — missing (recommended)")

    # Check data.yaml
    data_yaml = DATASET_DIR / "data.yaml"
    if data_yaml.exists():
        print(f"  ✅ data.yaml")
    else:
        print(f"  ❌ data.yaml — MISSING")
        all_good = False

    if not all_good:
        print(f"\n❌ Dataset verification failed.")
        sys.exit(1)

    # Fix paths and show stats
    fix_data_yaml()
    stats = count_dataset_stats()

    print(f"✅ Dataset verification passed!")
    return stats


def train(data_yaml_path: str = None):
    """Train YOLO11 on the strawberry dataset."""
    from ultralytics import YOLO

    if data_yaml_path is None:
        data_yaml_path = fix_data_yaml()

    # Detect device
    device = detect_device()
    TRAIN_CONFIG["device"] = device

    # Adjust batch size for device/model
    if device == "cpu":
        TRAIN_CONFIG["batch"] = 8  # Reduce batch on CPU
        print("   Reduced batch size to 8 for CPU training")

    print(f"\n{'='*60}")
    print(f"Training YOLO11 Strawberry Detector")
    print(f"{'='*60}")
    print(f"  Model:      {TRAIN_CONFIG['model']}")
    print(f"  Epochs:     {TRAIN_CONFIG['epochs']}")
    print(f"  Batch:      {TRAIN_CONFIG['batch']}")
    print(f"  ImgSize:    {TRAIN_CONFIG['imgsz']}")
    print(f"  Device:     {TRAIN_CONFIG['device']}")
    print(f"  Dataset:    {data_yaml_path}")
    print(f"  Optimizer:  {TRAIN_CONFIG['optimizer']}")
    print(f"  LR:         {TRAIN_CONFIG['lr0']} → {TRAIN_CONFIG['lr0'] * TRAIN_CONFIG['lrf']}")
    print(f"  Patience:   {TRAIN_CONFIG['patience']}")
    print(f"  Output:     {TRAIN_CONFIG['project']}/{TRAIN_CONFIG['name']}")
    print(f"{'='*60}\n")

    # Load pre-trained model
    model = YOLO(TRAIN_CONFIG["model"])

    # Separate model key from training kwargs
    train_kwargs = {k: v for k, v in TRAIN_CONFIG.items() if k != "model"}

    # Start timer
    start_time = time.time()

    # Train
    results = model.train(data=data_yaml_path, **train_kwargs)

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    # Check results
    run_name = TRAIN_CONFIG["name"]
    best_pt = RUNS_DIR / "detect" / run_name / "weights" / "best.pt"
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"✅ Training complete!")
        print(f"   Time:         {elapsed_min:.1f} minutes")
        print(f"   Best weights: {best_pt}")
        print(f"   Size:         {size_mb:.1f} MB")
        print(f"{'='*60}")

        # Copy best.pt to project root for easy access
        dest = PROJECT_DIR / "strawberry_yolo_best.pt"
        shutil.copy2(best_pt, dest)
        print(f"   Copied to: {dest}")
    else:
        print(f"\n❌ Training may have failed — best.pt not found at {best_pt}")

    return results


def test_model(weights_path: str = None):
    """Validate and test the trained model."""
    from ultralytics import YOLO

    if weights_path is None:
        # Search for weights in order of preference
        candidates = [
            PROJECT_DIR / "strawberry_yolo_best.pt",
            RUNS_DIR / "detect" / "strawberry_v2" / "weights" / "best.pt",
            RUNS_DIR / "detect" / "strawberry" / "weights" / "best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                weights_path = str(candidate)
                break

        if weights_path is None:
            print("❌ No trained weights found. Run --train first.")
            print("   Searched:")
            for c in candidates:
                print(f"     {c}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Testing model: {weights_path}")
    print(f"{'='*60}\n")

    model = YOLO(weights_path)
    device = detect_device()

    # Run validation
    data_yaml = fix_data_yaml()
    metrics = model.val(data=data_yaml, device=device)

    print(f"\n{'='*60}")
    print(f"Validation Results")
    print(f"{'='*60}")
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")

    # Per-class results
    if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
        print(f"\n  Per-class AP@50:")
        names = model.names
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = names[cls_idx]
            ap50 = metrics.box.ap50[i]
            print(f"    {cls_name}: {ap50:.4f}")

    print(f"{'='*60}")

    return metrics


def quick_inference(image_path: str, weights_path: str = None):
    """Run inference on a single image."""
    from ultralytics import YOLO

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    if weights_path is None:
        root_pt = PROJECT_DIR / "strawberry_yolo_best.pt"
        v2_pt = RUNS_DIR / "detect" / "strawberry_v2" / "weights" / "best.pt"
        if root_pt.exists():
            weights_path = str(root_pt)
        elif v2_pt.exists():
            weights_path = str(v2_pt)
        else:
            print("❌ No trained weights found. Run --train first.")
            sys.exit(1)

    model = YOLO(weights_path)
    device = detect_device()

    results = model(image_path, conf=0.25, device=device)

    for r in results:
        n_detections = len(r.boxes)
        print(f"\n{'='*60}")
        print(f"Inference: {image_path}")
        print(f"Detections: {n_detections}")
        print(f"{'='*60}")

        ripe_count = 0
        unripe_count = 0

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            name = model.names[cls]

            if "ripe" in name.lower() and "unripe" not in name.lower():
                ripe_count += 1
            elif "unripe" in name.lower():
                unripe_count += 1

            print(f"  {name}: {conf:.2f} at "
                  f"[{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")

        print(f"\n  Summary: {ripe_count} ripe, {unripe_count} unripe")

        # Save annotated image
        try:
            import cv2
            annotated = r.plot()
            out_dir = PROJECT_DIR / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"detected_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved annotated image: {out_path}")
        except ImportError:
            print("  ⚠️  OpenCV not available — skipping annotated image save")

    return results


def show_training_summary():
    """Show summary of the most recent training run."""
    # Check both possible run names
    for run_name in ["strawberry_v2", "strawberry"]:
        results_csv = RUNS_DIR / "detect" / run_name / "results.csv"
        if results_csv.exists():
            break
    else:
        print("❌ No training results found.")
        return

    import csv
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("❌ Results file is empty.")
        return

    last = rows[-1]
    print(f"\n{'='*60}")
    print(f"Training Summary — {run_name} ({len(rows)} epochs)")
    print(f"{'='*60}")

    # Print all metric columns
    for key in last:
        k = key.strip()
        v = last[key].strip()
        if any(m in k.lower() for m in ['map', 'precision', 'recall', 'loss']):
            print(f"  {k}: {v}")

    # Show best epoch
    try:
        map50_key = None
        for key in rows[0]:
            if 'map50' in key.strip().lower() and '95' not in key.strip().lower():
                map50_key = key
                break

        if map50_key:
            best_epoch = max(range(len(rows)),
                           key=lambda i: float(rows[i][map50_key].strip()))
            best_map50 = float(rows[best_epoch][map50_key].strip())
            print(f"\n  Best epoch: {best_epoch + 1} (mAP@50: {best_map50:.4f})")
    except (ValueError, KeyError):
        pass

    print(f"{'='*60}")


def export_model(weights_path: str = None, formats: list = None):
    """Export trained model to different formats for deployment."""
    from ultralytics import YOLO

    if weights_path is None:
        root_pt = PROJECT_DIR / "strawberry_yolo_best.pt"
        if root_pt.exists():
            weights_path = str(root_pt)
        else:
            print("❌ No trained weights found.")
            sys.exit(1)

    if formats is None:
        formats = ["onnx"]  # Default: ONNX for broad compatibility

    model = YOLO(weights_path)

    for fmt in formats:
        print(f"\n  Exporting to {fmt}...")
        try:
            export_path = model.export(format=fmt)
            print(f"  ✅ Exported: {export_path}")
        except Exception as e:
            print(f"  ❌ Export to {fmt} failed: {e}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 Strawberry Detector (ripe/unripe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --verify                    Check dataset is ready
  %(prog)s --train                     Train with default settings (YOLO11s)
  %(prog)s --train --model-size n      Train with Nano (faster, less accurate)
  %(prog)s --train --model-size m      Train with Medium (slower, more accurate)
  %(prog)s --train --epochs 150        Override epoch count
  %(prog)s --train --resume            Resume interrupted training
  %(prog)s --test                      Validate trained model
  %(prog)s --infer image.jpg           Run detection on an image
  %(prog)s --export                    Export to ONNX
  %(prog)s --verify --train --test     Full pipeline
        """
    )

    # Actions
    parser.add_argument("--verify", action="store_true",
                        help="Verify dataset structure and show statistics")
    parser.add_argument("--train", action="store_true",
                        help="Train the model")
    parser.add_argument("--test", action="store_true",
                        help="Run validation on trained model")
    parser.add_argument("--infer", type=str, metavar="IMAGE",
                        help="Run inference on a single image")
    parser.add_argument("--summary", action="store_true",
                        help="Show training summary from results.csv")
    parser.add_argument("--export", action="store_true",
                        help="Export model to ONNX format")

    # Training overrides
    parser.add_argument("--epochs", type=int,
                        help="Override number of training epochs")
    parser.add_argument("--batch", type=int,
                        help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Override image size (default: 640)")
    parser.add_argument("--model-size", type=str, choices=["n", "s", "m"],
                        default="s",
                        help="YOLO11 size: n(ano)/s(mall)/m(edium) [default: s]")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--weights", type=str,
                        help="Path to specific weights file for --test/--infer")

    args = parser.parse_args()

    # Apply overrides
    if args.epochs:
        TRAIN_CONFIG["epochs"] = args.epochs
    if args.batch:
        TRAIN_CONFIG["batch"] = args.batch
    if args.imgsz:
        TRAIN_CONFIG["imgsz"] = args.imgsz
    if args.model_size:
        TRAIN_CONFIG["model"] = f"yolo11{args.model_size}.pt"

    # No args → show help
    if not any([args.verify, args.train, args.test, args.infer,
                args.summary, args.export]):
        parser.print_help()
        print(f"\n{'='*60}")
        print("Quick Start:")
        print("  1. Place dataset in datasets/strawberry/")
        print("     (with train/valid/test splits and data.yaml)")
        print("  2. python train_strawberry_yolo.py --verify")
        print("  3. python train_strawberry_yolo.py --train")
        print("  4. python train_strawberry_yolo.py --test")
        print(f"{'='*60}")
        return

    # Execute actions in order
    if args.verify:
        verify_dataset()

    if args.train:
        if args.resume:
            # Find last checkpoint
            for run_name in ["strawberry_v2", "strawberry"]:
                last_pt = RUNS_DIR / "detect" / run_name / "weights" / "last.pt"
                if last_pt.exists():
                    TRAIN_CONFIG["model"] = str(last_pt)
                    print(f"📂 Resuming from {last_pt}")
                    break
            else:
                print("⚠️  No checkpoint found to resume — starting fresh")

        data_yaml = fix_data_yaml()
        train(data_yaml)

    if args.test:
        test_model(args.weights)

    if args.infer:
        quick_inference(args.infer, args.weights)

    if args.summary:
        show_training_summary()

    if args.export:
        export_model(args.weights)


if __name__ == "__main__":
    main()