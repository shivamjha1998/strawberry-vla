#!/usr/bin/env python3
"""
Train YOLO11 on Strawberry Detection Dataset
=============================================
Optimized for Mac Mini M4 (16GB RAM) using MPS acceleration.
Uses YOLO11n (better small-object detection than YOLOv8).
Default dataset: Harvesting Robot Strawberry Detection (1060 images).

Usage:
    # Step 1: Download dataset (needs free Roboflow API key)
    python train_strawberry_yolo.py --download --api-key YOUR_KEY

    # Step 2: Train
    python train_strawberry_yolo.py --train

    # Step 3: Validate & test
    python train_strawberry_yolo.py --test

    # All in one
    python train_strawberry_yolo.py --download --api-key YOUR_KEY --train --test
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "datasets" / "strawberry"
RUNS_DIR = PROJECT_DIR / "runs"
BEST_WEIGHTS = RUNS_DIR / "detect" / "strawberry" / "weights" / "best.pt"

# Training config optimized for Mac Mini M4 16GB
# Using YOLO11n — better small-object detection than YOLOv8, well-tested on MPS
TRAIN_CONFIG = {
    "model": "yolo11n.pt",       # YOLO11 Nano - best balance for M4 16GB
    "epochs": 80,                 # Good balance for fine-tuning
    "batch": 16,                  # Safe for 16GB RAM with nano model
    "imgsz": 640,                 # Standard YOLO input size
    "device": "mps",              # Apple Silicon GPU
    "workers": 4,                 # M4 has efficient cores
    "patience": 20,               # Early stopping if no improvement
    "save_period": 10,            # Save checkpoint every 10 epochs
    "project": str(RUNS_DIR / "detect"),
    "name": "strawberry",
    "exist_ok": True,
    "pretrained": True,           # Transfer learning from COCO
    "optimizer": "AdamW",
    "lr0": 0.001,                 # Lower LR for fine-tuning
    "lrf": 0.01,                  # Final LR fraction
    "cos_lr": True,               # Cosine LR schedule
    "mosaic": 1.0,                # Mosaic augmentation (great for small objects)
    "close_mosaic": 10,           # Turn off mosaic for last 10 epochs
    "hsv_h": 0.015,               # Hue augmentation
    "hsv_s": 0.7,                 # Saturation augmentation
    "hsv_v": 0.4,                 # Value augmentation
    "flipud": 0.1,                # Vertical flip (strawberries can hang)
    "fliplr": 0.5,                # Horizontal flip
    "degrees": 10,                # Slight rotation
    "scale": 0.5,                 # Scale augmentation
    "verbose": True,
}

# ─── Dataset Options ─────────────────────────────────────────────────────────
# We provide multiple dataset options - pick the best one for your needs

DATASETS = {
    # Option 1: Harvesting Robot dataset (RECOMMENDED)
    # 1060 images, 4 classes, made for harvesting robots, v9 (well-iterated)
    "harvesting": {
        "workspace": "harvesting-robot-datasets",
        "project": "strawberry-detection-msf0m",
        "version": 9,
        "description": "Harvesting Robot - strawberry-ripe/unripe (1060 images, v9) ⭐ BEST",
    },
    # Option 2: Strawberry ripeness detection (ripe/unripe/flower)
    "ripeness": {
        "workspace": "strawberries",
        "project": "strawberry-detect",
        "version": 1,
        "description": "Strawberry ripeness (ripe/unripe/flower) - 943 images",
    },
    # Option 3: Strawberry with maturity levels
    "maturity": {
        "workspace": "matt-lucky-f7mch",
        "project": "strawberry-k4gtp",
        "version": 1,
        "description": "Strawberry maturity levels (ripe/unripe) - 943 images",
    },
}

DEFAULT_DATASET = "harvesting"


def download_dataset(api_key: str, dataset_name: str = DEFAULT_DATASET):
    """Download strawberry dataset from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system(f"{sys.executable} -m pip install roboflow --break-system-packages -q")
        from roboflow import Roboflow

    ds_info = DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"Downloading: {ds_info['description']}")
    print(f"{'='*60}\n")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ds_info["workspace"]).project(ds_info["project"])
    version = project.version(ds_info["version"])

    # Download in YOLOv8 format
    dataset = version.download("yolov8", location=str(DATASET_DIR))

    # Verify download
    data_yaml = DATASET_DIR / "data.yaml"
    if data_yaml.exists():
        print(f"\n✅ Dataset downloaded to: {DATASET_DIR}")
        print(f"   data.yaml: {data_yaml}")

        # Show dataset stats
        import yaml
        with open(data_yaml) as f:
            config = yaml.safe_load(f)
        print(f"   Classes: {config.get('names', 'unknown')}")
        print(f"   NC: {config.get('nc', 'unknown')}")

        # Count images
        for split in ["train", "valid", "test"]:
            split_dir = DATASET_DIR / split / "images"
            if split_dir.exists():
                count = len(list(split_dir.glob("*")))
                print(f"   {split}: {count} images")
    else:
        print("❌ Download failed - data.yaml not found")
        sys.exit(1)

    return dataset


def fix_data_yaml():
    """Fix data.yaml paths to be absolute (Roboflow sometimes uses relative paths)."""
    import yaml

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print("❌ data.yaml not found. Run --download first.")
        sys.exit(1)

    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    # Fix paths to be absolute
    for key in ["train", "val", "valid", "test"]:
        if key in config:
            path = config[key]
            if not os.path.isabs(path):
                # Make path absolute relative to dataset dir
                abs_path = str(DATASET_DIR / path)
                config[key] = abs_path

    # Roboflow uses "val" but sometimes "valid" - normalize
    if "valid" in config and "val" not in config:
        config["val"] = config.pop("valid")

    with open(data_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Fixed data.yaml paths")
    return str(data_yaml)


def train(data_yaml_path: str = None):
    """Train YOLOv8 on the strawberry dataset."""
    from ultralytics import YOLO

    if data_yaml_path is None:
        data_yaml_path = fix_data_yaml()

    print(f"\n{'='*60}")
    print(f"Training YOLO11 Strawberry Detector")
    print(f"{'='*60}")
    print(f"  Model:    {TRAIN_CONFIG['model']}")
    print(f"  Epochs:   {TRAIN_CONFIG['epochs']}")
    print(f"  Batch:    {TRAIN_CONFIG['batch']}")
    print(f"  ImgSize:  {TRAIN_CONFIG['imgsz']}")
    print(f"  Device:   {TRAIN_CONFIG['device']}")
    print(f"  Dataset:  {data_yaml_path}")
    print(f"{'='*60}\n")

    # Check MPS availability
    try:
        import torch
        if not torch.backends.mps.is_available():
            print("⚠️  MPS not available, falling back to CPU")
            TRAIN_CONFIG["device"] = "cpu"
        else:
            print("✅ MPS (Apple Silicon GPU) available")
    except Exception:
        print("⚠️  Could not check MPS, trying anyway...")

    # Load pre-trained model
    model = YOLO(TRAIN_CONFIG["model"])

    # Train
    results = model.train(
        data=data_yaml_path,
        **{k: v for k, v in TRAIN_CONFIG.items() if k != "model"}
    )

    # Check results
    best_pt = RUNS_DIR / "detect" / "strawberry" / "weights" / "best.pt"
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"✅ Training complete!")
        print(f"   Best weights: {best_pt}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"{'='*60}")

        # Copy best.pt to project root for easy access
        dest = PROJECT_DIR / "strawberry_yolo_best.pt"
        shutil.copy2(best_pt, dest)
        print(f"   Copied to: {dest}")
    else:
        print("❌ Training may have failed - best.pt not found")

    return results


def test_model(weights_path: str = None):
    """Validate and test the trained model."""
    from ultralytics import YOLO

    if weights_path is None:
        # Try project root copy first, then runs dir
        root_pt = PROJECT_DIR / "strawberry_yolo_best.pt"
        if root_pt.exists():
            weights_path = str(root_pt)
        elif BEST_WEIGHTS.exists():
            weights_path = str(BEST_WEIGHTS)
        else:
            print("❌ No trained weights found. Run --train first.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Testing model: {weights_path}")
    print(f"{'='*60}\n")

    model = YOLO(weights_path)

    # Validation metrics
    data_yaml = fix_data_yaml()
    metrics = model.val(data=data_yaml, device="mps")

    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")
    print(f"{'='*60}")

    # Show per-class results
    if hasattr(metrics.box, 'ap_class_index'):
        print(f"\nPer-class AP@50:")
        names = model.names
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = names[cls_idx]
            ap50 = metrics.box.ap50[i]
            print(f"  {cls_name}: {ap50:.4f}")

    return metrics


def quick_inference(image_path: str, weights_path: str = None):
    """Run quick inference on a single image to verify the model works."""
    from ultralytics import YOLO
    import cv2

    if weights_path is None:
        root_pt = PROJECT_DIR / "strawberry_yolo_best.pt"
        if root_pt.exists():
            weights_path = str(root_pt)
        else:
            weights_path = str(BEST_WEIGHTS)

    model = YOLO(weights_path)
    results = model(image_path, conf=0.25, device="mps")

    for r in results:
        print(f"\nDetections in {image_path}:")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            name = model.names[cls]
            print(f"  {name}: {conf:.2f} at [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")

        # Save annotated image
        annotated = r.plot()
        out_path = PROJECT_DIR / "outputs" / f"yolo_test_{Path(image_path).stem}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved: {out_path}")

    return results


def show_training_summary():
    """Show summary of training results."""
    results_csv = RUNS_DIR / "detect" / "strawberry" / "results.csv"
    if not results_csv.exists():
        print("No training results found.")
        return

    import csv
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if rows:
        last = rows[-1]
        print(f"\n{'='*60}")
        print(f"Training Summary ({len(rows)} epochs completed)")
        print(f"{'='*60}")
        # Column names vary by ultralytics version, try common ones
        for key in last:
            k = key.strip()
            if any(m in k.lower() for m in ['map', 'precision', 'recall', 'loss']):
                print(f"  {k}: {last[key].strip()}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 Strawberry Detector")
    parser.add_argument("--download", action="store_true", help="Download dataset from Roboflow")
    parser.add_argument("--api-key", type=str, help="Roboflow API key (free at roboflow.com)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=DATASETS.keys(), help="Which dataset to use")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Run validation/testing")
    parser.add_argument("--infer", type=str, help="Run inference on an image")
    parser.add_argument("--summary", action="store_true", help="Show training summary")

    # Training overrides
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--model-size", type=str, choices=["n", "s", "m"],
                        default="n", help="YOLO11 size: n(ano)/s(mall)/m(edium)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--weights", type=str, help="Path to weights file")

    args = parser.parse_args()

    # Apply overrides
    if args.epochs:
        TRAIN_CONFIG["epochs"] = args.epochs
    if args.batch:
        TRAIN_CONFIG["batch"] = args.batch
    if args.model_size:
        TRAIN_CONFIG["model"] = f"yolo11{args.model_size}.pt"

    # No args = show help
    if not any([args.download, args.train, args.test, args.infer, args.summary]):
        parser.print_help()
        print(f"\n{'='*60}")
        print("Quick Start:")
        print("  1. Get free API key at https://app.roboflow.com/settings/api")
        print("  2. python train_strawberry_yolo.py --download --api-key YOUR_KEY")
        print("  3. python train_strawberry_yolo.py --train")
        print("  4. python train_strawberry_yolo.py --test")
        print(f"\nAvailable datasets:")
        for name, info in DATASETS.items():
            marker = " (default)" if name == DEFAULT_DATASET else ""
            print(f"  {name}: {info['description']}{marker}")
        print(f"{'='*60}")
        return

    if args.download:
        if not args.api_key:
            print("❌ Need --api-key for download. Get one free at https://app.roboflow.com/settings/api")
            sys.exit(1)
        download_dataset(args.api_key, args.dataset)

    if args.train:
        if args.resume:
            last_pt = RUNS_DIR / "detect" / "strawberry" / "weights" / "last.pt"
            if last_pt.exists():
                TRAIN_CONFIG["model"] = str(last_pt)
                print(f"Resuming from {last_pt}")
        data_yaml = fix_data_yaml()
        train(data_yaml)

    if args.test:
        test_model(args.weights)

    if args.infer:
        quick_inference(args.infer, args.weights)

    if args.summary:
        show_training_summary()


if __name__ == "__main__":
    main()
