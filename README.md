# Strawberry VLA

**Vision-Language-Action system for strawberry greenhouse operations.**

Fine-tuned YOLO11s object detection + fine-tuned Qwen 3 VL 8B vision-language analysis + RGB ripeness scoring, running locally on Apple Silicon.

---

## Architecture

Two-stage pipeline optimized for real-time inference:

| Stage | Model | Speed | Output |
|-------|-------|-------|--------|
| 1 — Detection | YOLO11s (fine-tuned, 4,536 images) | ~10ms / frame | Bounding boxes + ripe/unripe classification |
| 2a — Ripeness | RGB color analysis | Instant | Ripeness percentage, harvest readiness |
| 2b — Detailed | Qwen 3 VL 8B + LoRA adapter | ~10s / crop | Ripeness, health, harvest recommendation |
| 2c — Disease | Qwen 3 VL 8B + LoRA adapter | ~10s / frame | Disease assessment (powdery mildew, anthracnose) |

## Capabilities

1. **Strawberry coordinate detection** — YOLO11s detects strawberry bounding boxes from greenhouse video/images
2. **Ripeness assessment via RGB** — HSV/RGB color analysis with 5-stage classification (green, white, turning, ripe, overripe)
3. **Disease detection** — Fine-tuned Qwen 3 VL identifies powdery mildew, anthracnose, botrytis, and leaf spot
4. **3D coordinate calibration** — Methodology documented (multi-camera stereo calibration), pending hardware setup

## Features

- **YouTube video analysis** — paste a URL, extract frames, detect and analyze strawberries
- **Image upload** — single image detection and analysis
- **Fine-tuned models** — both YOLO and Qwen trained on domain-specific strawberry data
- **Bilingual UI** — English and Japanese language support
- **Fully offline** — runs entirely on local Apple Silicon hardware

## Requirements

- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4) recommended
- **RAM**: 16GB+ (Qwen 3 VL 8B uses ~5GB in 4-bit quantization + 333MB adapter)
- **Python**: 3.10+

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd strawberry-vla

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Model Setup

**YOLO11** — Train the strawberry detection model:

```bash
python train_strawberry_yolo.py --verify --train --test
# The trained model saves to strawberry_yolo_best.pt
```

**Qwen 3 VL** — The base model is downloaded automatically on first use via `mlx-vlm`. The LoRA adapter must be placed at `output/v2-20260308-003328/checkpoint-861-mlx/`.

## Usage

### Web Demo

```bash
# Standard launch
python demo_app.py

# Pre-load models before UI starts
python demo_app.py --preload

# Create a public shareable URL
python demo_app.py --share
```

Open http://localhost:7860 in your browser.

### CLI Detection

```bash
# Fast mode (YOLO + RGB only)
python strawberry_detector.py --image frame.jpg --visualize

# Detailed mode (+ Qwen VL per-crop analysis)
python strawberry_detector.py --image frame.jpg --detailed --visualize

# With fine-tuned adapter
python strawberry_detector.py --image frame.jpg --detailed --adapter-path output/v2-20260308-003328/checkpoint-861-mlx --visualize

# With disease detection
python strawberry_detector.py --frames frames/VIDEO/ --disease --visualize
```

## Project Structure

```
strawberry-vla/
├── demo_app.py                 # Gradio web UI
├── strawberry_detector.py      # Core detection pipeline (YOLO + Qwen VL + RGB)
├── train_strawberry_yolo.py    # YOLO11 fine-tuning script
├── compare_models.py           # Base vs fine-tuned model comparison
├── convert_adapter_to_mlx.py   # PEFT to MLX adapter converter
├── requirements.txt            # Python dependencies
├── report.md                   # Phase 1 report
├── report_phase2.md            # Phase 2 report (fine-tuning results)
├── locales/
│   ├── en.json                 # English translations
│   └── ja.json                 # Japanese translations
├── favicon.png                 # Browser tab icon
├── strawberry_yolo_best.pt     # Fine-tuned YOLO11s model (not in git)
└── output/                     # Fine-tuning outputs (not in git)
    └── v2-20260308-003328/
        ├── checkpoint-861/     # PEFT LoRA adapter
        └── checkpoint-861-mlx/ # MLX LoRA adapter (for Apple Silicon)
```

## Model Performance

**YOLO11s (Strawberry Detection) — Phase 2**
- Trained on 4,536 labeled images (Roboflow dataset, 2 classes: ripe/unripe)
- mAP@50 = 93.3% | mAP@50-95 = 77.5%
- Precision = 85.3% | Recall = 87.7%
- Inference: ~10ms per frame on Apple Silicon

**Qwen 3 VL 8B (Fine-tuned) — Phase 2**
- LoRA fine-tuned on 2,291 strawberry Q&A samples (RTX PRO 6000, 96GB)
- 4-bit quantized via MLX (~5GB + 333MB adapter)
- Inference: ~10s per crop on Apple Silicon

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|-----------|------------|-------------|
| Ripeness Accuracy | 48% | 68% | +42% |
| Harvest Recommendation | 26% | 70% | +169% |
| Health Assessment | 88% | 94% | +7% |
| ROUGE-L | 0.175 | 0.444 | +153% |

## Reports

- [Phase 1 Report](report.md) — System architecture, YOLO training (1,060 images), RGB analysis, disease detection methodology
- [Phase 2 Report](report_phase2.md) — Fine-tuning results (YOLO 4,536 images + Qwen 3 VL LoRA), model comparison, recommendations

## Tech Stack

| Component | Library |
|-----------|---------|
| Object Detection | Ultralytics YOLO11 |
| Vision-Language Model | Qwen 3 VL 8B via mlx-vlm |
| Fine-Tuning | ms-swift 4.0 + PEFT (LoRA) |
| Video Download | yt-dlp |
| Frame Extraction | OpenCV |
| Web UI | Gradio |
| Dataset Management | Roboflow |
| Inference (Apple Silicon) | MLX / mlx-vlm |

## License

This project is for research and educational purposes.