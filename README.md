# Strawberry VLA

**Vision-Language-Action system for strawberry greenhouse operations.**

YOLO11 object detection + Qwen 2.5 VL vision-language analysis + RGB ripeness scoring, running locally on Apple Silicon.

---

## Architecture

Two-stage pipeline optimized for real-time inference:

| Stage | Model | Speed | Output |
|-------|-------|-------|--------|
| 1 — Detection | YOLO11 (fine-tuned) | ~10ms / frame | Bounding boxes + classification |
| 2a — Ripeness | RGB color analysis | Instant | Ripeness percentage, harvest readiness |
| 2b — Detailed | Qwen 2.5 VL 7B (optional) | ~10-30s / crop | Per-strawberry natural language analysis |
| 2c — Disease | Qwen 2.5 VL 7B (optional) | ~10-30s / frame | Disease assessment |

## Features

- **YouTube video analysis** — paste a URL, extract frames, detect and analyze strawberries
- **Image upload** — single image detection and analysis
- **Ripeness scoring** — HSV/RGB color analysis with harvest readiness classification
- **Disease detection** — optional Qwen VL-powered disease assessment
- **Bilingual UI** — English and Japanese language support
- **Pastel web UI** — clean Gradio interface with custom styling

## Requirements

- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4) recommended
- **RAM**: 16GB+ (Qwen VL 7B uses ~4.5GB in 4-bit quantization)
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

**YOLO11** — Download or train the strawberry detection model:

```bash
# Option A: Train from scratch (requires Roboflow API key)
python train_strawberry_yolo.py --download --api-key YOUR_KEY --train --test

# The trained model saves to strawberry_yolo_best.pt
```

**Qwen 2.5 VL** — Downloaded automatically on first use via `mlx-vlm`. The 7B 4-bit quantized model (~4.5GB) is fetched from HuggingFace.

## Usage

### Web Demo

```bash
# Standard launch
python demo_app.py

# Pre-load models before UI starts
python demo_app.py --preload

# Create a public shareable URL
python demo_app.py --share

# Or use the launch script
./launch_demo.sh
```

Open http://localhost:7860 in your browser.

### CLI Detection

```bash
# Fast mode (YOLO + RGB only)
python strawberry_detector.py --image frame.jpg --visualize

# Detailed mode (+ Qwen VL per-crop analysis)
python strawberry_detector.py --image frame.jpg --detailed --visualize

# Process a directory of frames
python strawberry_detector.py --frames frames/VIDEO/ --visualize

# With disease detection
python strawberry_detector.py --frames frames/VIDEO/ --disease --visualize
```

### Video Ingestion

```bash
# Download YouTube video and extract frames
python video_ingest.py --url "https://youtube.com/watch?v=..." --fps 1

# Extract from curated video list
python video_ingest.py --from-list --fps 1

# Extract from local video file
python video_ingest.py --local path/to/video.mp4 --fps 2
```

## Project Structure

```
strawberry-vla/
├── demo_app.py                 # Gradio web UI
├── strawberry_detector.py      # Core detection pipeline (YOLO + Qwen VL + RGB)
├── video_ingest.py             # YouTube download & frame extraction
├── train_strawberry_yolo.py    # YOLO11 fine-tuning script
├── preview_frames.py           # Frame preview utility
├── launch_demo.sh              # Quick-start shell script
├── requirements.txt            # Python dependencies
├── locales/
│   ├── en.json                 # English translations
│   └── ja.json                 # Japanese translations
├── favicon.png                 # Browser tab icon
└── strawberry_yolo_best.pt     # Fine-tuned YOLO11 model (not in git)
```

## Model Performance

**YOLO11 (Strawberry Detection)**
- Trained on 1,060 labeled images (Roboflow dataset)
- mAP@50 = 87.1%
- Inference: ~10ms per frame on Apple Silicon

**Qwen 2.5 VL 7B**
- 4-bit quantized via MLX (~4.5GB)
- Inference: ~10-30s per crop on Apple Silicon
- Used for detailed per-strawberry analysis and disease detection

## Tech Stack

| Component | Library |
|-----------|---------|
| Object Detection | Ultralytics YOLO11 |
| Vision-Language Model | Qwen 2.5 VL 7B via mlx-vlm |
| Video Download | yt-dlp |
| Frame Extraction | OpenCV |
| Web UI | Gradio |
| Dataset Management | Roboflow |

## License

This project is for research and educational purposes.
