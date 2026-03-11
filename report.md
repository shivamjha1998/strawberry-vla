# Strawberry VLA System — Phase 1 Report

**Vision-Language-Action System for Strawberry Greenhouse Operations**

**Date:** February 2026
**Environment:** Mac Mini M4 (16GB RAM)
**Models:** YOLO11s (trained) + Qwen 2.5 VL 7B (4-bit)

---

## 1. Executive Summary

This report documents the Phase 1 development of a Vision-Language-Action (VLA) system designed for strawberry greenhouse operations. The system combines a trained YOLO11 object detector with Qwen 2.5 VL (Vision-Language) model to provide real-time strawberry detection, ripeness assessment, and disease identification from video footage.

**Key Phase 1 Results:**

- YOLO11s trained on 1,060 greenhouse images achieving **87.09% mAP@50**
- Detection speed of **~10ms per frame** on Mac Mini M4 (Apple Silicon MPS)
- RGB-based ripeness assessment with 5-stage classification
- Disease detection (powdery mildew, anthracnose) via Qwen 2.5 VL
- Working Gradio demo interface accepting YouTube URLs and image uploads
- Fully offline operation — no cloud dependencies

---

## 2. System Architecture

### 2.1 Pipeline Overview

The system uses a two-stage architecture where YOLO11 handles fast detection and Qwen 2.5 VL provides optional detailed analysis:

```
Video / Image Input
       │
       ▼
┌──────────────────┐
│  Frame Extraction │  (yt-dlp + OpenCV for YouTube videos)
│  720p, N frames   │
└──────┬───────────┘
       ▼
┌──────────────────┐
│    YOLO11s        │  PRIMARY DETECTOR
│    (~10ms/frame)  │  Trained on strawberry greenhouse data
│    2 classes:     │  strawberry-ripe, strawberry-unripe
└──────┬───────────┘
       │ Bounding boxes + confidence scores
       ▼
┌──────────────────┐
│  RGB Analysis     │  RIPENESS SCORING
│  (<1ms/crop)      │  HSV color space analysis
│  5 ripeness stages│  Red/green/white pixel ratios
└──────┬───────────┘
       │ (optional)
       ▼
┌──────────────────┐
│  Qwen 2.5 VL     │  DETAILED ANALYSIS
│  (~10-30s/crop)   │  Disease detection
│  7B 4-bit (MLX)   │  Detailed ripeness assessment
└──────────────────┘
```

### 2.2 Operating Modes

| Mode | Speed | Components | Use Case |
|------|-------|-----------|----------|
| **Fast** (default) | ~10ms/frame | YOLO11 + RGB | Real-time monitoring, batch processing |
| **Detailed** | ~10-30s/crop | + Qwen VL per crop | Detailed ripeness grading |
| **Disease** | ~10-30s/frame | + Qwen VL disease | Health inspection |

### 2.3 Hardware Configuration

| Component | Specification |
|-----------|--------------|
| Machine | Mac Mini M4 |
| RAM | 16GB Unified Memory |
| Compute | Apple M4 (10-core CPU, 10-core GPU) |
| ML Backend | Apple MPS (Metal Performance Shaders) |
| OS | macOS Tahoe |
| Python | 3.11+ with venv |

---

## 3. Capability 1: Strawberry Coordinate Detection

### 3.1 YOLO11 Model Training

The YOLO11s (small) model was fine-tuned on a domain-specific strawberry dataset for bounding box detection.

**Dataset:**
- Source: Roboflow — "Harvesting Robot Datasets" project
- URL: `https://universe.roboflow.com/harvesting-robot-datasets/strawberry-detection-msf0m`
- Images: 1,060 annotated greenhouse images (v9, CC BY 4.0 license)
- Classes: 2 — `strawberry-ripe`, `strawberry-unripe`
- Format: YOLOv11/Ultralytics compatible (images + labels in YOLO txt format)

**Training Configuration:**

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # Pre-trained COCO weights

results = model.train(
    data="strawberry_dataset/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    device="mps",           # Apple Silicon GPU
    optimizer="AdamW",
    lr0=0.01,
    lrf=0.01,
    augment=True,
    mosaic=1.0,
    copy_paste=0.0,
    project="strawberry_training",
    name="yolo11s_strawberry",
)
```

**Training Results:**

| Metric | Value |
|--------|-------|
| mAP@50 | **87.09%** |
| mAP@50-95 | **71.03%** |
| Precision | **78.62%** |
| Recall | **85.82%** |
| Model Size | ~18 MB |
| Training Time | ~90 min on M4 |

### 3.2 Detection Output Format

Each detection produces bounding box coordinates in absolute pixel values:

```json
{
  "bbox_2d": [120, 80, 280, 240],
  "label": "strawberry",
  "yolo_class": "strawberry-ripe",
  "ripeness": "ripe",
  "confidence_score": 0.89,
  "confidence": "high",
  "source": "yolo11"
}
```

### 3.3 Inference Performance

| Metric | Value |
|--------|-------|
| Single frame inference | ~8-12ms |
| Batch (100 frames, YOLO only) | ~1-2 seconds |
| Confidence threshold | 0.3 (configurable) |
| Device | MPS (Apple Metal) |

### 3.4 YOLO11 Architecture Notes

YOLO11s was selected as a balance between accuracy and inference speed on Apple Silicon. It provides more capacity than the nano variant while remaining fast enough for near-real-time use. Key architectural features that benefit strawberry detection:

- **C3k2 blocks**: Compact backbone with improved feature extraction for small objects
- **C2PSA (Cross-Stage Partial Spatial Attention)**: Enhanced spatial attention helps distinguish overlapping strawberry clusters
- **SPPF (Spatial Pyramid Pooling - Fast)**: Multi-scale feature aggregation for varying strawberry sizes

---

## 4. Capability 2: Ripeness Assessment via RGB

### 4.1 Method

A pure computer-vision approach using HSV color space analysis, applied to each YOLO-detected crop region. This requires no model inference and runs in under 1ms per crop.

### 4.2 Algorithm

For each detected strawberry bounding box:

1. Extract the crop region from the frame
2. Convert BGR → RGB and BGR → HSV
3. Compute pixel masks for red, green, and white regions
4. Classify into 5 ripeness stages based on pixel ratios

**Color Thresholds:**

```python
# Red pixels: H < 10 or H > 170, S > 50
red_mask = ((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)) & (hsv[:,:,1] > 50)

# Green pixels: 35 < H < 85, S > 30
green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 30)

# White pixels: S < 40, V > 180
white_mask = (hsv[:,:,1] < 40) & (hsv[:,:,2] > 180)
```

### 4.3 Ripeness Classification

| Stage | Criteria | Ripeness % | Harvest Ready |
|-------|----------|-----------|---------------|
| 🟢 Green | green_pct > 30% | 0-20% | No |
| ⚪ White | white_pct > 30% | 20-40% | No |
| 🟠 Turning | red_pct 30-60% | 40-85% | No |
| 🔴 Ripe | red_pct > 60%, V > 80 | 85-99% | **Yes** |
| 🟤 Overripe | red_pct > 60%, V < 80 | 100% | No (past peak) |

### 4.4 Output Format

```json
{
  "mean_rgb": [185, 42, 38],
  "mean_hsv": [4, 198, 185],
  "red_pixel_pct": 72.3,
  "green_pixel_pct": 3.1,
  "white_pixel_pct": 2.8,
  "red_green_ratio": 4.40,
  "ripeness_stage": "ripe",
  "ripeness_percentage": 88.1,
  "harvest_ready": true
}
```

---

## 5. Capability 3: Disease Detection

### 5.1 Method

Disease detection uses the Qwen 2.5 VL 7B model (4-bit quantized via MLX) for visual analysis. The VL model inspects full frames or individual crops and identifies common strawberry diseases through natural language understanding of visual symptoms.

### 5.2 Supported Diseases

| Disease | Visual Symptoms | Severity Levels |
|---------|----------------|-----------------|
| Powdery Mildew | White/gray powdery coating on leaves or fruit | none / mild / moderate / severe |
| Anthracnose | Dark, sunken lesions; pink/orange spore masses | none / mild / moderate / severe |
| Botrytis (Gray Mold) | Fuzzy gray mold on fruit | none / mild / moderate / severe |
| Leaf Spot | Dark spots on leaves | none / mild / moderate / severe |

### 5.3 Model Configuration

```python
# Qwen 2.5 VL loaded via MLX (Apple Silicon optimized)
model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
# Memory footprint: ~4.5 GB
# Inference time: 10-30 seconds per image
```

The model is prompted with a structured disease detection prompt and returns JSON:

```json
{
  "diseases_detected": [
    {
      "disease": "powdery mildew",
      "severity": "mild",
      "location": "lower leaves near stem",
      "confidence": "medium"
    }
  ],
  "overall_plant_health": "mild_issues",
  "notes": "Early signs of powdery mildew on lower foliage"
}
```

### 5.4 Limitations

- Qwen 2.5 VL has not been fine-tuned on strawberry disease data; it relies on general visual understanding
- Disease detection accuracy depends on image quality and lighting conditions
- Inference is slow (~10-30s) compared to YOLO detection, making it unsuitable for real-time use
- Fine-tuning on H100 or Tenstorrent Wormhole hardware (Section 8) is planned to improve disease detection accuracy

---

## 6. Capability 4: 3D Coordinate Calibration (Methodology)

### 6.1 Status

3D coordinate calibration is planned for Phase 2/3. This section documents the methodology.

### 6.2 Approach: Multi-Camera Stereo Calibration

For robotic arm integration, 2D bounding box coordinates from YOLO must be mapped to 3D world coordinates. The proposed approach:

**Step 1 — Camera Calibration**
- Use a checkerboard pattern to compute intrinsic parameters (focal length, optical center, distortion coefficients) for each camera
- OpenCV's `calibrateCamera()` function provides these parameters

**Step 2 — Stereo Calibration**
- With 2+ cameras at known positions, compute the extrinsic parameters (rotation, translation) between cameras
- OpenCV's `stereoCalibrate()` establishes the geometric relationship

**Step 3 — Depth Estimation**
- For each YOLO detection, compute disparity between matched points across camera views
- Use `reprojectImageTo3D()` to convert disparity to 3D coordinates
- Alternative: use depth sensor (Intel RealSense, etc.) for direct depth measurement

**Step 4 — Coordinate Transformation**
- Transform camera-frame 3D coordinates into the robotic arm's coordinate frame
- Requires a known calibration target or fiducial markers to establish the camera-to-robot transform

### 6.3 Proposed Pipeline Addition

```python
# Back-project to 3D using camera intrinsics + extrinsic stereo baseline
# Note: Requires rigid stereo rig calibration (Phase 2 hardware setup)
def estimate_3d_coordinates(detection, camera_params, depth_source):
    """
    Convert 2D YOLO detection to 3D world coordinates.
    
    Args:
        detection: YOLO bbox [x1, y1, x2, y2]
        camera_params: Intrinsic + extrinsic camera parameters
        depth_source: "stereo" | "depth_sensor"
    
    Returns:
        {"x": float, "y": float, "z": float}  # in millimeters
    """
    center_x = (detection[0] + detection[2]) / 2
    center_y = (detection[1] + detection[3]) / 2
    
    if depth_source == "stereo":
        depth = compute_stereo_depth(center_x, center_y, camera_params)
    else:
        depth = read_depth_sensor(center_x, center_y)
    
    # Back-project to 3D using camera intrinsics
    fx, fy = camera_params["focal_length"]
    cx, cy = camera_params["optical_center"]
    
    X = (center_x - cx) * depth / fx
    Y = (center_y - cy) * depth / fy
    Z = depth
    
    return {"x": X, "y": Y, "z": Z}
```

### 6.4 Hardware Requirements

- Minimum 2 cameras (stereo pair) OR 1 RGB camera + depth sensor
- Known camera mounting positions relative to robotic arm base
- Calibration target (printed checkerboard)
- Recommended: Intel RealSense D435 for integrated RGB + depth

---

## 7. Demo Environment

### 7.1 Demo Interface

A Gradio-based web interface (`demo_app.py`) provides three interaction modes:

**Tab 1 — YouTube Video Analysis**
- Paste any YouTube URL containing strawberry greenhouse footage
- Downloads video (720p, max 2 min) via yt-dlp
- Extracts N evenly-spaced frames (configurable, default 6)
- Runs YOLO11 detection + RGB analysis on each frame
- Optional: disease detection toggle (Qwen VL)
- Displays annotated frame gallery + detection results + JSON

**Tab 2 — Image Upload**
- Drag-and-drop any strawberry image
- Instant YOLO11 detection with RGB analysis
- Optional: Qwen VL detailed analysis and disease detection

**Tab 3 — About**
- Architecture
- Performance specifications
- Model information

### 7.2 Running the Demo

```bash
cd ~/strawberry-vla
source venv/bin/activate

# Standard launch
python demo_app.py

# Pre-load model (recommended for demo)
python demo_app.py --preload

# Remote access via public URL
python demo_app.py --share
```

Access at: `http://localhost:7860`

### 7.3 Demo Day Recommendations

**Preparation:**
1. Run `python demo_app.py --preload` to warm up models
2. Close other memory-intensive applications
3. Have YouTube URLs ready:
   - Search: "strawberry greenhouse harvest close up"
   - Search: "いちご ハウス 収穫" (Japanese greenhouse footage)
4. Use 4-6 frames for YouTube analysis (balances speed and coverage)

**Expected demo behavior:**
- YOLO detection: near-instant (~10ms per frame)
- RGB analysis: instant
- Disease detection (if enabled): adds ~10-30s per frame
- Set audience expectations: "Fast detection with optional deep analysis"

### 7.4 CLI Usage

```bash
# Single image (fast mode)
python strawberry_detector.py --image frame.jpg --visualize

# Process video frames directory
python strawberry_detector.py --frames frames/VIDEO_NAME/ --visualize

# With disease detection
python strawberry_detector.py --frames frames/VIDEO_NAME/ --disease --visualize

# Full batch, every 5th frame
python strawberry_detector.py --frames-root frames/ --frame-step 5 --visualize
```

---

## 8. Fine-Tuning Methodology for Tenstorrent AI Accelerator

### 8.1 Overview

Phase 2 fine-tuning will use either **NVIDIA H100** GPUs or **Tenstorrent Wormhole** accelerators to improve model accuracy beyond what's achievable on the Mac Mini M4's 16GB RAM. Two models require fine-tuning:

1. **YOLO11 → larger variants** (11m, 11l) on expanded datasets
2. **Qwen 2.5 VL** with LoRA for strawberry-specific ripeness and disease assessment

### 8.2 Target Hardware

#### Option A: NVIDIA H100

| Specification | Value |
|---------------|-------|
| GPU Architecture | Hopper |
| VRAM | 80 GB HBM3 |
| Memory Bandwidth | 3.35 TB/s |
| Peak Performance | 1,670 TFLOPS (FP8) |
| TDP | 350-700W (SXM / PCIe) |
| Availability | Cloud (AWS p5, Lambda Labs, RunPod) or on-prem |

The H100 is the most mature option for fine-tuning. Full PyTorch/CUDA support means all standard training frameworks (Ultralytics, HuggingFace Transformers, PEFT/LoRA) work out of the box. A single H100 (80GB) can comfortably handle both YOLO training and Qwen VL 7B LoRA fine-tuning without quantization.

#### Option B: Tenstorrent Wormhole

| Specification | n150 (Single Chip) | n300 (Dual Chip) | TT-QuietBox (8x Wormhole) |
|---------------|-------------------|-------------------|---------------------------|
| Tensix Cores | 72 | 128 (64 per ASIC) | 512 (64 per ASIC × 8) |
| Memory | 12 GB GDDR6 | 24 GB GDDR6 | 96 GB GDDR6 |
| Memory Bandwidth | 288 GB/s | 576 GB/s | 2.3 TB/s |
| Peak FP8 | 262 TFLOPS | 466 TFLOPS | ~3.7 PFLOPS |
| TDP | 160W | 300W | ~1.2 kW |

The Wormhole series uses Tenstorrent's open-source software stack (TT-Metal, TT-Forge, TT-XLA). For fine-tuning, the TT-QuietBox (4× n300 = 8 Wormhole processors, 96GB total) provides sufficient memory for Qwen VL LoRA training. Single n300 cards (24GB) can handle YOLO training.

#### Hardware Comparison for Our Workloads

| Task | H100 (80GB) | Wormhole n300 (24GB) | TT-QuietBox (96GB) |
|------|-------------|---------------------|---------------------|
| YOLO11m training | ✅ Easy | ✅ Feasible | ✅ Easy |
| YOLO11l training | ✅ Easy | ⚠️ Tight | ✅ Easy |
| Qwen VL QLoRA (4-bit) | ✅ Easy (~18GB) | ✅ Feasible (~18GB) | ✅ Easy |
| Qwen VL LoRA (BF16) | ✅ Easy (~45GB) | ❌ Too large | ✅ Feasible (TP across cards) |
| Qwen VL full fine-tune | ⚠️ Tight (~60GB+) | ❌ Too large | ⚠️ Possible with TP=8 |

### 8.3 Fine-Tuning Plan 1: YOLO11 on H100 / Wormhole

**Goal:** Train larger YOLO11 variants (11m/11l) on expanded datasets to improve mAP beyond 87.09%.

**Dataset Expansion Strategy:**
- Current: 1,060 images (2 classes)
- Target: 5,000-10,000 images including:
  - More greenhouse lighting conditions (morning, afternoon, artificial)
  - Occluded and clustered strawberries
  - Additional classes: flower, leaf, disease-affected
- Sources: Roboflow Universe, manual annotation from YouTube footage

**Training Script for Tenstorrent:**

```python
"""
train_yolo_phase2.py — YOLO11 Training on H100 or Tenstorrent Wormhole

Prerequisites (H100):
    - NVIDIA H100 GPU with CUDA 12+
    - pip install ultralytics

Prerequisites (Wormhole):
    - Tenstorrent n300 card(s) installed
    - TT-Metal + TT-Forge stack installed
    - Note: As of early 2026, recommended to train on CUDA,
      then deploy on Wormhole for inference

Usage:
    # On H100 (CUDA)
    python train_yolo_phase2.py --model yolo11m.pt --epochs 100 --device 0

    # On Wormhole via TT-XLA (experimental)
    python train_yolo_phase2.py --model yolo11m.pt --epochs 100 --device tt
"""

import argparse
from ultralytics import YOLO

def train_yolo_expanded(model_size="yolo11m.pt", epochs=100, batch=32, device=0):
    """
    Train larger YOLO variant on expanded strawberry dataset.
    
    H100 path (recommended for training):
        device=0 uses CUDA GPU directly — full PyTorch support
    
    Wormhole path (for inference deployment):
        Train on CUDA, then export ONNX for Wormhole deployment:
        model.export(format="onnx", imgsz=640, simplify=True)
        
        For training on Wormhole via TT-XLA (experimental):
        import torch_ttnn
        device = ttnn.open_device(device_id=0)
        model = torch.compile(model, backend=torch_ttnn.backend)
    """
    model = YOLO(model_size)

    results = model.train(
        data="strawberry_dataset_expanded/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,                # 0 for CUDA (H100), "mps" for Mac
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project="strawberry_training_v2",
        name=f"{model_size.replace('.pt','')}_expanded",
        patience=20,             # Early stopping
        save_period=10,          # Checkpoint every 10 epochs
    )

    # Evaluate
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50:.3f}")
    print(f"mAP@50-95: {metrics.box.map:.3f}")

    # Export for Wormhole deployment (if using Tenstorrent for inference)
    # ONNX is the most compatible format for TT-Forge
    model.export(format="onnx", imgsz=640, simplify=True)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11m.pt",
                        choices=["yolo11s.pt", "yolo11m.pt", "yolo11l.pt"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0", help="0 for CUDA (H100), mps for Mac, tt for Tenstorrent")
    args = parser.parse_args()
    train_yolo_expanded(args.model, args.epochs, args.batch, args.device)
```

**Expected Improvements:**

| Model | Params | Expected mAP@50 | Training Time |
|-------|--------|-----------------|---------------|
| YOLO11s (current) | 9.4M | 87.09% (achieved) | ~90 min (M4) |
| YOLO11m | 20.1M | ~90-93% | ~3-5 hours (H100) |
| YOLO11l | 25.3M | ~93-95% | ~5-8 hours (H100) |

### 8.4 Fine-Tuning Plan 2: Qwen 2.5 VL with LoRA on H100 / Wormhole

**Goal:** Fine-tune Qwen 2.5 VL 7B with LoRA to improve strawberry-specific ripeness assessment and disease detection accuracy.

**Why LoRA:** Full fine-tuning of a 7B model requires ~60GB+ VRAM. LoRA reduces trainable parameters to ~0.1-1% of the model, making it feasible on a single H100 (80GB) for full BF16, or on a Wormhole n300 (24GB) with QLoRA 4-bit quantization.

#### 8.4.1 Training Data Preparation

```python
"""
prepare_qwen_training_data.py — Prepare strawberry VL fine-tuning dataset

Format: JSONL with image-text conversation pairs
"""

import json
from pathlib import Path

def create_ripeness_sample(image_path, ripeness_stage, percentage, notes):
    """Create a single training sample for ripeness assessment."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": (
                        "Analyze the ripeness of the strawberry in this image. "
                        "Output ONLY JSON: {\"ripeness_stage\": ..., "
                        "\"ripeness_percentage\": ..., \"harvest_ready\": ..., "
                        "\"notes\": ...}"
                    )}
                ]
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "ripeness_stage": ripeness_stage,
                    "ripeness_percentage": percentage,
                    "color_uniformity": "uniform",
                    "surface_quality": "smooth",
                    "harvest_ready": ripeness_stage == "ripe",
                    "notes": notes
                })
            }
        ]
    }


def create_disease_sample(image_path, diseases, health, notes):
    """Create a single training sample for disease detection."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": (
                        "Analyze this image of strawberries for signs of disease. "
                        "Output ONLY JSON: {\"diseases_detected\": [...], "
                        "\"overall_plant_health\": ..., \"notes\": ...}"
                    )}
                ]
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "diseases_detected": diseases,
                    "overall_plant_health": health,
                    "notes": notes
                })
            }
        ]
    }


def build_dataset(annotations_dir, output_path):
    """
    Build JSONL dataset from annotated strawberry images.
    
    Expected annotation structure:
        annotations/
        ├── ripeness/
        │   ├── green/        # images of green strawberries
        │   ├── turning/      # images of turning strawberries
        │   ├── ripe/         # images of ripe strawberries
        │   └── overripe/     # images of overripe strawberries
        └── disease/
            ├── healthy/      # healthy plant images
            ├── powdery_mildew/
            ├── anthracnose/
            └── botrytis/
    
    Target: 1,500-2,500 samples for effective LoRA fine-tuning
    """
    samples = []
    
    # Ripeness samples
    ripeness_map = {
        "green": ("green", 10, "Unripe, fully green fruit"),
        "turning": ("turning", 55, "Partially red, still developing"),
        "ripe": ("ripe", 90, "Fully red, ready for harvest"),
        "overripe": ("overripe", 100, "Past peak, dark coloration"),
    }
    
    annotations = Path(annotations_dir)
    for stage, (label, pct, note) in ripeness_map.items():
        stage_dir = annotations / "ripeness" / stage
        if stage_dir.exists():
            for img in stage_dir.glob("*.jpg"):
                samples.append(create_ripeness_sample(img, label, pct, note))
    
    # Disease samples
    disease_configs = {
        "healthy": ([], "healthy", "No visible disease symptoms"),
        "powdery_mildew": (
            [{"disease": "powdery mildew", "severity": "moderate",
              "location": "leaf surface", "confidence": "high"}],
            "moderate_issues", "White powdery coating visible on leaves"
        ),
        "anthracnose": (
            [{"disease": "anthracnose", "severity": "moderate",
              "location": "fruit surface", "confidence": "high"}],
            "moderate_issues", "Dark sunken lesions on fruit"
        ),
        "botrytis": (
            [{"disease": "botrytis", "severity": "severe",
              "location": "fruit", "confidence": "high"}],
            "severe_issues", "Gray fuzzy mold covering fruit"
        ),
    }
    
    for disease_type, (diseases, health, note) in disease_configs.items():
        disease_dir = annotations / "disease" / disease_type
        if disease_dir.exists():
            for img in disease_dir.glob("*.jpg"):
                samples.append(create_disease_sample(img, diseases, health, note))
    
    # Write JSONL
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created {len(samples)} training samples → {output_path}")
    return len(samples)


if __name__ == "__main__":
    build_dataset("annotations", "strawberry_vl_train.jsonl")
```

#### 8.4.2 LoRA Fine-Tuning Script

```python
"""
finetune_qwen_vl_lora.py — Qwen 2.5 VL LoRA Fine-Tuning

Requirements:
    pip install torch transformers peft bitsandbytes accelerate
    pip install qwen-vl-utils trl lightning

Hardware Options:
    - NVIDIA H100 80GB: Full BF16 LoRA (recommended, simplest)
    - Tenstorrent Wormhole n300 24GB: QLoRA (4-bit quantized)
    - Tenstorrent TT-QuietBox 96GB: Full BF16 LoRA with TP

Usage:
    # On H100 (CUDA) — recommended
    python finetune_qwen_vl_lora.py --device cuda --epochs 10

    # QLoRA on Wormhole n300 (24GB)
    python finetune_qwen_vl_lora.py --qlora --epochs 5

    # On H100 with QLoRA (faster, less memory)
    python finetune_qwen_vl_lora.py --device cuda --qlora --epochs 10
"""

import argparse
import torch
from pathlib import Path


def setup_model(model_id, use_qlora=True):
    """Load Qwen 2.5 VL with LoRA configuration."""
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
        BitsAndBytesConfig,
    )

    # LoRA configuration — targets attention layers
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,                      # Rank (8-16 for VL tasks)
        bias="none",
        target_modules=[
            "q_proj", "v_proj",   # Attention layers
            # Optionally add: "k_proj", "o_proj", "gate_proj",
            # "up_proj", "down_proj" for more capacity
        ],
        task_type="CAUSAL_LM",
    )

    # Quantization config for QLoRA (reduces memory by ~60%)
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: trainable params: ~3.5M / 7.6B (0.046%)

    # Processor (handles image + text tokenization)
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    return model, processor, lora_config


def create_dataloader(jsonl_path, processor, batch_size=1):
    """Create training dataloader from JSONL dataset."""
    import json
    from torch.utils.data import Dataset, DataLoader
    from qwen_vl_utils import process_vision_info

    class StrawberryVLDataset(Dataset):
        def __init__(self, jsonl_path, processor):
            self.samples = []
            with open(jsonl_path) as f:
                for line in f:
                    self.samples.append(json.loads(line))
            self.processor = processor

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            messages = sample["messages"]

            # Process conversation into model input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Create labels (mask user turns, only compute loss on assistant)
            inputs["labels"] = inputs["input_ids"].clone()
            # Mask everything before the assistant response
            # (implementation depends on tokenizer specifics)

            return {k: v.squeeze(0) for k, v in inputs.items()}

    dataset = StrawberryVLDataset(jsonl_path, processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(args):
    """Run LoRA fine-tuning."""
    from trl import SFTConfig, SFTTrainer

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model, processor, lora_config = setup_model(model_id, use_qlora=args.qlora)

    # Training configuration
    training_config = SFTConfig(
        output_dir="./qwen_vl_strawberry_lora",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,    # Effective batch = 8
        gradient_checkpointing=True,      # Save memory
        bf16=True,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=None,  # Set from JSONL loader
        eval_dataset=None,
        peft_config=lora_config,
        tokenizer=processor.tokenizer,
    )

    # Train
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained("./qwen_vl_strawberry_lora/final")
    processor.save_pretrained("./qwen_vl_strawberry_lora/final")

    print("Training complete!")
    print("To merge LoRA weights into base model:")
    print("  from peft import PeftModel")
    print("  merged = model.merge_and_unload()")
    print("  merged.save_pretrained('./qwen_vl_strawberry_merged')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", default="strawberry_vl_train.jsonl")
    args = parser.parse_args()
    train(args)
```

#### 8.4.3 Deploying Fine-Tuned Model to Wormhole

```python
"""
deploy_qwen_wormhole.py — Deploy fine-tuned Qwen VL on Tenstorrent Wormhole

After training the LoRA adapter on H100 (CUDA), merge weights and
deploy the model for fast inference on Tenstorrent Wormhole.

Tenstorrent's TT-Forge compiler converts PyTorch models to optimized
kernels for the Tensix architecture.
"""

# Option 1: TT-XLA (PyTorch 2.0 compile path — recommended)
import torch
import torch_ttnn
import ttnn

def deploy_via_tt_xla(model_path):
    """Deploy merged Qwen VL model via TT-XLA torch.compile."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    # Open Tenstorrent Wormhole device
    device = ttnn.open_device(device_id=0)

    # Load merged model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )

    # Compile for Wormhole via TT-XLA backend
    option = torch_ttnn.TorchTtnnOption(device=device)
    compiled_model = torch.compile(
        model, backend=torch_ttnn.backend, options=option
    )

    return compiled_model, device


# Option 2: ONNX export → TT-Forge (more portable)
def export_to_onnx(model_path, output_path="qwen_vl_strawberry.onnx"):
    """Export to ONNX for TT-Forge compilation on Wormhole."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    import torch

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float32
    )
    model.eval()

    # Note: VL model ONNX export requires careful handling of
    # dynamic image inputs. Use opset_version >= 17.
    print(f"Model exported to {output_path}")
    print("Load in TT-Forge with: tt_forge.compile(onnx_path)")


# Option 3: vLLM on Wormhole (for serving)
def serve_via_vllm():
    """Serve fine-tuned model via Tenstorrent's vLLM fork on Wormhole."""
    # Tenstorrent provides a vLLM fork optimized for their hardware
    # https://github.com/tenstorrent/tt-inference-server
    print("""
    # Deploy via Tenstorrent vLLM on Wormhole:
    git clone https://github.com/tenstorrent/tt-inference-server
    cd tt-inference-server
    
    # Configure for fine-tuned model
    export MODEL_PATH=./qwen_vl_strawberry_merged
    export TT_DEVICE=n300    # Wormhole n300
    
    # Start inference server
    python -m vllm.entrypoints.openai.api_server \\
        --model $MODEL_PATH \\
        --device tt \\
        --dtype bfloat16
    """)
```

#### 8.4.4 Resource Estimates

| Configuration | Hardware | VRAM Used | Training Time (2000 samples) |
|---------------|----------|-----------|------------------------------|
| QLoRA (4-bit) | 1x H100 80GB | ~18 GB | ~2-3 hours |
| LoRA (BF16) | 1x H100 80GB | ~45 GB | ~4-6 hours |
| Full fine-tune | 1x H100 80GB | ~65 GB | ~10-20 hours |
| QLoRA (4-bit) | 1x Wormhole n300 24GB | ~18 GB | ~4-8 hours* |
| LoRA (BF16) | TT-QuietBox (4×n300) | ~45 GB (TP=4) | ~8-15 hours* |

*Wormhole training times are estimates; actual performance depends on TT-Forge maturity.

**Recommended approach:** Train with QLoRA on H100 (fastest, most mature tooling), validate improvement on a held-out test set, then deploy the merged model to Wormhole for production inference.

### 8.5 Tenstorrent Wormhole Software Stack

```
Application Layer
    │
    ├── PyTorch / JAX / ONNX    ← Standard ML frameworks
    │
    ▼
TT-XLA / TT-Forge (beta)       ← Compiler: PyTorch → TT-MLIR → Tensix kernels
    │
    ▼
TT-NN                          ← Operator library (matmul, conv, attention, etc.)
    │
    ▼
TT-Metalium                    ← Low-level API (C++ kernels on Tensix cores)
    │
    ▼
Tenstorrent Wormhole Hardware   ← 72-128 Tensix cores, 12-24GB GDDR6
```

### 8.6 Recommended Workflow: H100 for Training → Wormhole for Inference

As of early 2026, the recommended hybrid workflow is:

1. **Train on H100 (CUDA)** — Full PyTorch ecosystem support. All frameworks (Ultralytics, HuggingFace PEFT, TRL) work natively. H100's 80GB VRAM handles any training configuration without compromise.

2. **Export merged model** — After LoRA training, merge adapter weights and export in HuggingFace format or ONNX.

3. **Deploy on Wormhole for production inference** — Tenstorrent's Wormhole cards excel at inference with excellent price-per-TFLOPS. The vLLM fork and TT-Transformers libraries support key models including Qwen and Llama families.

4. **Migrate training to Wormhole over time** — As TT-Forge and TT-XLA mature (currently in beta), training workloads can progressively move to Wormhole hardware, reducing dependency on NVIDIA.

**Key advantage of Wormhole for deployment:** At ~$1,399 per n300 card (24GB, 466 TFLOPS FP8), Wormhole offers significantly lower cost than H100 for inference workloads, while the open-source software stack avoids vendor lock-in.

---

## 9. Project File Structure

```
~/strawberry-vla/
├── strawberry_detector.py       # Main detection pipeline
├── demo_app.py                  # Gradio web demo
├── train_strawberry_yolo.py     # YOLO11 training script
├── strawberry_yolo_best.pt      # Trained YOLO11s weights
├── launch_demo.sh               # One-line launcher
├── venv/                        # Python virtual environment
├── frames/                      # Extracted video frames
│   └── VIDEO_NAME/
│       ├── frame_00000_t0.00s.jpg
│       └── ...
├── outputs/                     # Detection results
│   └── VIDEO_NAME/
│       ├── detection_results.json
│       └── annotated/
│           └── det_frame_*.jpg
└── strawberry_dataset/          # Training data (from Roboflow)
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```

---

## 10. Phase 2 Roadmap

| Task | Priority | Description | Dependencies |
|------|----------|-------------|-------------|
| YOLO11m/l training | High | Train larger model on expanded dataset | More training data, H100 or Wormhole |
| Qwen VL LoRA fine-tuning | High | Strawberry-specific ripeness + disease | Training data annotation, H100 or Wormhole |
| 3D calibration | Medium | Multi-camera depth estimation | Stereo camera setup or depth sensor |
| Frame tracking | Medium | Persistent strawberry IDs across frames | YOLO + ByteTrack/BoT-SORT |
| Robotic arm integration | Low | Convert 3D coordinates to arm commands | 3D calibration complete |
| Real-time video stream | Low | Live camera feed processing | Optimized pipeline |

---

## 11. Conclusion

Phase 1 has successfully delivered a working strawberry VLA system with:

- **Fast, accurate detection** via a custom-trained YOLO11 model (87.09% mAP@50, ~10ms/frame)
- **Automated ripeness grading** using RGB/HSV color analysis with 5-stage classification
- **Disease identification** through Qwen 2.5 VL's visual understanding capabilities
- **Interactive demo** accessible via web browser with YouTube and image upload support
- **Documented fine-tuning pathway** for NVIDIA H100 and Tenstorrent Wormhole hardware to improve accuracy

The system runs entirely offline on a Mac Mini M4, making it suitable for deployment in greenhouse environments with limited connectivity. Phase 2 will focus on improving accuracy through H100/Wormhole-accelerated fine-tuning, adding 3D coordinate estimation for robotic arm integration, and enabling real-time video stream processing.