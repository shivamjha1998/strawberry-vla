# Strawberry VLA (Vision-Language-Action) Project Plan

## Project Overview

Building a VLA system for strawberry greenhouse operations using YOLO (object detection) + Qwen 2.5 VL (vision-language model). The system should detect strawberries from video, assess ripeness, detect diseases, and calibrate 3D coordinates for robotic harvesting.

**Hardware**: Mac Mini (demo/inference), Tenstorrent AI accelerator (fine-tuning)
**Models**: YOLO (detection), Qwen 2.5 VL (vision-language reasoning)

---

## Phase 1: Friday Demo & Report (Week 1) — DEADLINE: Friday

### Goal
Working demo on Mac Mini: feed strawberry video from YouTube → model detects strawberries and provides basic analysis. Accompanying report documenting architecture and fine-tuning methodology.

### Tasks

#### 1.1 Environment Setup — Mac Mini
- [ ] Install Python environment (conda/venv)
- [ ] Install Qwen 2.5 VL on Mac Mini (options: MLX, llama.cpp, or transformers with MPS)
- [ ] Verify Qwen 2.5 VL inference works on Mac Mini (test with sample image)
- [ ] Install YOLO (ultralytics) and verify detection works
- [ ] Document hardware specs, model size chosen, quantization level
- **Notes**: Qwen 2.5 VL 7B is the realistic size for Mac Mini. Consider 4-bit quantization if memory is tight.

#### 1.2 Video Ingestion Pipeline
- [ ] Build YouTube video download script (yt-dlp)
- [ ] Frame extraction module (ffmpeg or OpenCV, configurable FPS)
- [ ] Collect 3-5 good strawberry greenhouse YouTube videos for testing
- [ ] Store extracted frames in organized directory structure

#### 1.3 Strawberry Detection Pipeline
- [ ] YOLO strawberry detection on extracted frames (pre-trained or fine-tuned on fruit dataset)
- [ ] Crop detected strawberry regions from frames
- [ ] Pass cropped regions + full frame to Qwen 2.5 VL with structured prompts
- [ ] Qwen VL outputs: coordinates, count, basic ripeness description
- [ ] Simple visualization: annotated frames with bounding boxes + labels

#### 1.4 Demo Interface
- [ ] Simple script or Gradio/Streamlit UI that takes a YouTube URL as input
- [ ] Displays annotated frames with detection results
- [ ] Shows Qwen VL analysis text output alongside images

#### 1.5 Report
- [ ] System architecture diagram
- [ ] Model selection rationale (why YOLO + Qwen 2.5 VL)
- [ ] Mac Mini performance benchmarks (inference speed, memory usage)
- [ ] Fine-tuning methodology write-up (see Phase 3)
- [ ] Tenstorrent accelerator plan (see Phase 4)
- [ ] Roadmap for Phase 2-4

---

## Phase 2: Core Capabilities (Weeks 2-3)

### 2.1 Strawberry Coordinate Detection (Refined)
- [ ] Fine-tune YOLO on strawberry-specific dataset
- [ ] Source/create labeled strawberry dataset (Roboflow, custom labeling)
- [ ] Improve bounding box accuracy for greenhouse conditions
- [ ] Handle occlusion, clusters, varying lighting
- [ ] Output structured coordinate data (JSON format)

### 2.2 Ripeness Assessment via RGB
- [ ] Extract RGB values from detected strawberry regions
- [ ] Define ripeness classification levels:
  - Green (unripe)
  - White/turning (intermediate)
  - Light red (nearly ripe)
  - Deep red (ripe)
  - Overripe/dark
- [ ] Build color histogram analysis per detected strawberry
- [ ] Train simple classifier OR use Qwen VL with calibrated prompts
- [ ] Validate against ground truth data
- [ ] Output ripeness score + confidence per strawberry

### 2.3 Disease Detection
- [ ] Source strawberry disease image datasets:
  - Powdery mildew (white fuzzy patches on leaves/fruit)
  - Anthracnose (dark sunken lesions on fruit)
- [ ] Approach A: Fine-tune YOLO to detect disease regions
- [ ] Approach B: Use Qwen VL with disease-specific prompts
- [ ] Approach C: Combine both (YOLO detects regions, VL classifies disease type)
- [ ] Build evaluation metrics (precision, recall, F1)
- [ ] Test on real greenhouse footage

---

## Phase 3: 3D Coordinate Calibration (Weeks 3-4)

### 3.1 Multi-Camera Setup
- [ ] Define camera placement strategy for greenhouse
- [ ] Camera intrinsic calibration (checkerboard method, OpenCV)
- [ ] Camera extrinsic calibration (relative positioning)
- [ ] Stereo calibration if using stereo pairs

### 3.2 3D Reconstruction
- [ ] Implement triangulation from multi-view detections
- [ ] Match strawberry detections across camera views (feature matching or ID tracking)
- [ ] Convert 2D pixel coordinates → 3D world coordinates
- [ ] Accuracy validation (measure real positions vs computed positions)
- [ ] Output format: (x, y, z) in real-world units (mm/cm)

### 3.3 Integration with VLA
- [ ] Feed 3D coordinates into action planning module
- [ ] Map coordinates to robot arm workspace (future)

---

## Phase 4: Fine-Tuning & Optimization (Weeks 4-5)

### 4.1 Fine-Tuning Methodology (documented in Phase 1 report)
- [ ] Dataset preparation pipeline (collection, labeling, augmentation)
- [ ] YOLO fine-tuning procedure (Ultralytics training config)
- [ ] Qwen 2.5 VL fine-tuning procedure (LoRA/QLoRA approach)
- [ ] Evaluation framework (mAP for detection, accuracy for classification)

### 4.2 Tenstorrent Accelerator Setup
- [ ] Tenstorrent SDK installation and environment setup
- [ ] Model conversion to Tenstorrent-compatible format
- [ ] Fine-tuning pipeline on Tenstorrent hardware
- [ ] Benchmark: Tenstorrent vs GPU training speed/cost
- [ ] Document any limitations or workarounds

### 4.3 Accuracy Improvement Iterations
- [ ] Collect more training data from greenhouse footage
- [ ] Iterative fine-tuning cycles
- [ ] A/B testing different model configurations
- [ ] Target accuracy metrics per capability

---

## Tech Stack

| Component | Tool/Library | Purpose |
|---|---|---|
| Object Detection | Ultralytics YOLO (v8/v11) | Strawberry bounding box detection |
| Vision-Language Model | Qwen 2.5 VL (7B) | Scene understanding, ripeness/disease analysis |
| Mac Inference | MLX / llama.cpp / transformers | Run Qwen VL on Mac Mini |
| Video Download | yt-dlp | YouTube video ingestion |
| Frame Extraction | OpenCV / ffmpeg | Video → frames |
| 3D Calibration | OpenCV | Camera calibration, triangulation |
| Demo UI | Gradio or Streamlit | Interactive demo interface |
| Dataset Management | Roboflow / LabelImg | Image labeling and dataset versioning |
| Fine-Tuning | PyTorch + LoRA (PEFT) | Model fine-tuning |
| Accelerator | Tenstorrent SDK | Hardware-accelerated fine-tuning |

---

## Key Resources & Datasets

- [ ] Strawberry detection datasets (Roboflow has several)
- [ ] Strawberry disease image datasets (Kaggle, PlantVillage)
- [ ] YouTube strawberry greenhouse videos (list URLs here as collected)
- [ ] Qwen 2.5 VL documentation: https://github.com/QwenLM/Qwen2.5-VL
- [ ] Tenstorrent SDK documentation: (add link when found)

---

## Progress Log

Use this section to track what's been done. Copy this entire document into new LLM sessions as context.

### Format
```
[DATE] — TASK — STATUS — NOTES
```

### Log
```
[YYYY-MM-DD] — Project plan created — DONE — Initial plan drafted
```

---

## Current Status Summary

**Last Updated**: [DATE]

**Phase 1 Status**: NOT STARTED
- Environment: Not set up
- Pipeline: Not built
- Demo: Not built
- Report: Not written

**Blockers**: None yet

**Next Action**: Start with Mac Mini environment setup (Task 1.1)

---

## LLM Context Handoff Instructions

When starting a new LLM chat session, paste this entire document and add:

```
I'm working on this Strawberry VLA project. Here's the project plan and current status.
[Paste this document]

Current status: [describe what's done and what you're working on next]
Last thing I was working on: [specific task/problem]
I need help with: [specific request]
```

This gives the new LLM instance full context to continue helping effectively.
