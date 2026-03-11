# Strawberry VLA System — Phase 2 Report

**Fine-Tuning YOLO + Qwen 3 VL for Strawberry Agricultural Analysis**

**Date:** March 2026
**Author:** Strawberry VLA Team
**Environment:** Mac Mini M4 (16GB) for inference, RunPod RTX PRO 6000 (96GB) for Qwen fine-tuning, Mac Mini M4 (MPS) for YOLO training

---

## 1. Executive Summary

Phase 2 focused on improving system accuracy through two major fine-tuning efforts:

1. **YOLO11s re-training** on a significantly larger curated strawberry dataset (4,536 images, up from 1,060 in Phase 1)
2. **Qwen 3 VL 8B fine-tuning** with LoRA adapters using a custom-generated strawberry Q&A dataset (2,291 training samples)

**Key Achievements:**

| Achievement | Detail |
|-------------|--------|
| YOLO dataset | 4x larger (1,060 -> 4,536 images), 2-class detection (ripe/unripe) |
| Qwen model upgrade | Qwen 2.5 VL 7B -> Qwen 3 VL 8B (fine-tuned) |
| Ripeness accuracy | +41.7% improvement (48% -> 68%) over base model |
| Harvest recommendation | +169.2% improvement (26% -> 70%) over base model |
| Health assessment | +6.8% improvement (88% -> 94%) over base model |
| Response quality (ROUGE-L) | +153.1% improvement (0.175 -> 0.444) |
| Answer conciseness | 44.8% shorter responses (161 -> 89 words avg) |
| Inference speed | 10.5% faster (11.3s -> 10.1s per query) |

---

## 2. YOLO11s Fine-Tuning (Object Detection)

### 2.1 Dataset

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Source | Roboflow (Harvesting Robot Datasets) | Roboflow (expanded curated dataset) |
| Total images | 1,060 | **4,536** |
| Classes | 2 (ripe, unripe) | 2 (ripe, unripe) |
| Format | YOLOv11/Ultralytics | YOLOv11/Ultralytics |
| Splits | train/valid/test | train/valid/test |

### 2.2 Training Configuration

```python
TRAIN_CONFIG = {
    "model": "yolo11s.pt",           # YOLO11 Small (pre-trained on COCO)
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "mps",                 # Apple Silicon GPU
    "patience": 25,                  # Early stopping
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,                     # Final LR = lr0 * lrf
    "cos_lr": True,                  # Cosine annealing schedule
    # Augmentation
    "mosaic": 1.0,
    "close_mosaic": 15,
    "hsv_h": 0.015,                  # Minimal hue shift (preserve ripeness color cues)
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "flipud": 0.3,
    "fliplr": 0.5,
    "degrees": 15,
    "scale": 0.4,
    "mixup": 0.1,
    "copy_paste": 0.1,
}
```

**Key design decisions:**
- Hue augmentation kept very low (0.015) to avoid confusing ripe/unripe color distinctions
- Mosaic augmentation enabled for spatial context, disabled for last 15 epochs for clean fine-tuning
- AdamW optimizer with low initial LR (0.001) for transfer learning from COCO pre-trained weights

### 2.3 Training Results

| Metric | Phase 1 (1,060 images) | Phase 2 (4,536 images) |
|--------|----------------------|----------------------|
| mAP@50 | 87.09% | 93.3% |
| mAP@50-95 | 71.03% | 77.5% |
| Precision | 78.62% | 85.3% |
| Recall | 85.82% | 87.7% |
| Model size | ~18 MB | ~18 MB |
| Inference speed | ~10ms/frame | ~10ms/frame |
| Training device | Mac Mini M4 (MPS) | Mac Mini M4 (MPS) |

### 2.4 YOLO Output

The trained model (`strawberry_yolo_best.pt`, 18.3 MB) detects strawberries with bounding boxes classified as `ripe` or `unripe`:

```json
{
    "bbox_2d": [120, 80, 280, 240],
    "label": "strawberry",
    "yolo_class": "strawberry-ripe",
    "confidence_score": 0.89
}
```

---

## 3. Q&A Dataset Generation Pipeline

### 3.1 Pipeline Overview

A custom pipeline was built to generate high-quality training data for Qwen fine-tuning, bridging YOLO detection output with vision-language model training:

```
YOLO Detection on Greenhouse Images
        |
        v
Crop 18,256 Individual Strawberry Regions
        |
        v
Generate Q&A Pairs using Base Qwen Model
(ripeness, health, harvest recommendations)
        |
        v
Human Review & Quality Filtering
        |
        v
Train/Val Split (90/10)
    2,291 train / 255 val
```

### 3.2 Dataset Statistics

| Parameter | Value |
|-----------|-------|
| Source images | Greenhouse footage (YOLO detections) |
| Individual crop images | 18,256 |
| Q&A pairs generated | 2,546 |
| Training samples | 2,291 |
| Validation samples | 255 |
| Train/Val ratio | 90% / 10% |
| Data format | JSONL (ms-swift/ShareGPT compatible) |

### 3.3 Data Format

Each training sample follows the ShareGPT multi-turn format with system prompt, user query (with image reference), and expert-level assistant response:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are an expert agricultural scientist specializing in strawberry cultivation and disease detection. Analyze the strawberry image and provide a concise assessment covering ripeness, color, health status, and harvest recommendation."
        },
        {
            "role": "user",
            "content": "<image>Is this strawberry showing any signs of disease? Describe what you see."
        },
        {
            "role": "assistant",
            "content": "Based on the image provided, here is the assessment:\n\n1. **Ripeness level:** Green (unripe)\n2. **Estimated dominant color:** Pale green to yellowish-green\n3. **Health status:** Healthy...\n4. **Harvest recommendation:** Wait..."
        }
    ],
    "images": ["crops/image_det0.jpg"]
}
```

### 3.4 Question Types

The dataset covers diverse agricultural analysis questions:
- Ripeness assessment and grading
- Disease symptom identification
- Harvest readiness evaluation
- Health status analysis
- Quality concern identification

---

## 4. Qwen 3 VL 8B Fine-Tuning

### 4.1 Model Selection

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-VL-8B-Instruct |
| Architecture | Qwen3-VL (Vision-Language) |
| Total parameters | 8,854.4M (8.85B) |
| Trainable parameters | 87.3M (0.99% of total) |
| Fine-tuning method | LoRA (Low-Rank Adaptation) |
| Framework | ms-swift 4.0 (wraps HuggingFace PEFT) |
| Precision | bfloat16 |

### 4.2 LoRA Configuration

```
Tuner type:        LoRA (PEFT v0.18.1)
Rank (r):          32
Alpha:             64
Dropout:           0.05
Bias:              none
Target modules:    all-linear (7 modules per transformer layer)
                   - q_proj, k_proj, v_proj, o_proj
                   - gate_proj, up_proj, down_proj
Frozen components: Vision encoder (ViT), Aligner
Trainable:         Language model linear layers only
```

**Why LoRA?**
- Only 0.99% of model parameters are trained (87.3M out of 8,854.4M)
- Dramatically reduces GPU memory requirements
- Adapter weights are only ~333 MB (vs 16+ GB for full model)
- Can be swapped or removed without modifying the base model

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA RTX PRO 6000 (96GB VRAM) |
| Cloud platform | RunPod |
| Epochs | 3 |
| Batch size | 1 (per device) |
| Gradient accumulation | 8 (effective batch = 8) |
| Learning rate | 1e-4 |
| LR scheduler | Cosine with warmup |
| Warmup ratio | 5% |
| Optimizer | AdamW (fused) |
| Weight decay | 0.1 |
| Max sequence length | 4,096 tokens |
| Gradient checkpointing | Enabled |
| Total steps | 861 |
| Training time | ~25 minutes 32 seconds |
| Peak GPU memory | 19.67 GiB |

### 4.4 Training Progression

The model showed rapid convergence with consistent improvement across all 3 epochs:

**Loss Curve:**

| Checkpoint | Train Loss | Eval Loss | Token Accuracy |
|-----------|------------|-----------|----------------|
| Step 1 | 1.901 | — | 61.1% |
| Step 50 | 0.599 | — | 81.5% |
| Step 100 | 0.499 | 0.464 | 83.2% / 84.5% (eval) |
| Step 200 | 0.388 | 0.400 | 86.0% / 86.3% (eval) |
| Step 300 | 0.293 | 0.370 | 89.7% / 87.2% (eval) |
| Step 400 | 0.290 | 0.348 | 89.1% / 88.0% (eval) |
| Step 500 | 0.307 | 0.330 | 89.0% / 88.6% (eval) |
| Step 600 | 0.235 | 0.325 | 91.2% / 88.8% (eval) |
| Step 700 | 0.241 | 0.318 | 91.3% / 89.1% (eval) |
| Step 800 | 0.229 | 0.313 | 91.7% / 89.3% (eval) |
| Step 861 (final) | 0.208 | **0.312** | 92.2% / **89.3%** (eval) |

**Key observations:**
- Loss dropped from 1.90 -> 0.21 (89% reduction)
- Token accuracy improved from 61.1% -> 92.2% (training), 84.5% -> 89.3% (validation)
- No signs of overfitting (eval loss continued decreasing through all 3 epochs)
- Best checkpoint was the final one (step 861)

### 4.5 Saved Checkpoints

| Checkpoint | Steps | Eval Loss | Location |
|-----------|-------|-----------|----------|
| checkpoint-700 | 700 | 0.318 | `output/v2-20260308-003328/checkpoint-700/` |
| checkpoint-800 | 800 | 0.313 | `output/v2-20260308-003328/checkpoint-800/` |
| **checkpoint-861** | **861** | **0.312** | `output/v2-20260308-003328/checkpoint-861/` |
| checkpoint-861-mlx | 861 | — | `output/v2-20260308-003328/checkpoint-861-mlx/` |

### 4.6 Adapter Conversion (PEFT -> MLX)

To run the fine-tuned model locally on Mac Mini M4, the PyTorch PEFT adapter was converted to MLX-VLM format:

| Aspect | PEFT (PyTorch) | MLX-VLM |
|--------|---------------|---------|
| File | `adapter_model.safetensors` | `adapters.safetensors` |
| Config key for rank | `r` | `rank` |
| Config alpha | `lora_alpha` (64) | `alpha` (2.0 = 64/32) |
| Weight names | `lora_A.weight`, `lora_B.weight` | `A`, `B` |
| Weight shapes | Transposed | Native MLX layout |
| Key prefix | `base_model.model.model.language_model.` | `language_model.model.` |

Conversion handled by `convert_adapter_to_mlx.py`.

---

## 5. Model Comparison: Base vs Fine-Tuned

### 5.1 Evaluation Setup

| Parameter | Value |
|-----------|-------|
| Test samples | 50 (from validation set) |
| Base model | `mlx-community/Qwen3-VL-8B-Instruct-4bit` |
| Fine-tuned model | Same base + LoRA adapter (`checkpoint-861-mlx`) |
| Inference device | Mac Mini M4 (16GB, MLX) |
| Quantization | 4-bit (W4A16) |

### 5.2 Results Summary

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|-----------|------------|-------------|
| **Ripeness Accuracy** | 48.0% | **68.0%** | +20.0 pp (+41.7%) |
| **Harvest Recommendation** | 26.0% | **70.0%** | +44.0 pp (+169.2%) |
| **Health Assessment** | 88.0% | **94.0%** | +6.0 pp (+6.8%) |
| **Response Structure** | 95.5% | **100.0%** | +4.5 pp (+4.7%) |
| **ROUGE-L** | 0.175 | **0.444** | +0.269 (+153.1%) |
| **Keyword F1** | 0.183 | **0.474** | +0.291 (+159.4%) |
| **Avg Response Length** | 161 words | **89 words** | -72 words (-44.8%) |
| **Avg Inference Time** | 11.26s | **10.07s** | -1.18s (-10.5%) |

### 5.3 Analysis

**Largest improvements:**
- **Harvest Recommendation (+169%):** The base model rarely matched domain-specific harvest language ("wait", "harvest now"). Fine-tuning taught the model our exact terminology and decision criteria.
- **ROUGE-L (+153%) and Keyword F1 (+159%):** Fine-tuned answers closely match ground truth phrasing, using consistent agricultural terminology.
- **Ripeness Accuracy (+42%):** The model learned our 5-stage ripeness classification (green, white, turning, ripe, overripe) rather than using generic descriptions.

**Response quality:**
- Fine-tuned model produces **shorter, more focused** answers (89 vs 161 words)
- **100% structure compliance** — always outputs in the expected numbered format
- Faster inference due to generating fewer tokens

**Already strong (minimal change):**
- Health assessment was already good (88% -> 94%) since disease symptoms are visually distinctive

### 5.4 Comparison Visualizations

Six comparison charts were generated in `comparison_results/`:
1. `accuracy_comparison.png` — Bar chart of accuracy metrics
2. `quality_comparison.png` — Bar chart of text quality metrics
3. `radar_comparison.png` — Radar plot of all metrics
4. `ripeness_breakdown.png` — Per-stage ripeness accuracy
5. `rouge_scatter.png` — Per-sample ROUGE-L scatter plot
6. `summary_table.png` — Summary table visualization

---

## 6. System Integration

### 6.1 Updated Architecture

```
Video / Image Input
       |
       v
+------------------+
|  YOLO11s          |  OBJECT DETECTION (fine-tuned, 4,536 images)
|  ~10ms/frame      |  2 classes: ripe, unripe
|  strawberry_yolo_best.pt (18.3 MB)
+--------+---------+
         |
         v
+------------------+
|  RGB Analysis     |  RIPENESS SCORING (rule-based, <1ms)
|  5 ripeness stages|  HSV color space analysis
+--------+---------+
         | (optional)
         v
+------------------+
|  Qwen 3 VL 8B    |  DETAILED ANALYSIS (fine-tuned)
|  + LoRA adapter   |  Domain-specific ripeness, health, harvest
|  ~10s/crop        |  4-bit quantized via MLX (~5GB)
|  Adapter: 333 MB  |
+------------------+
```

### 6.2 Demo Application

The Gradio demo (`demo_app.py`) now uses the fine-tuned Qwen 3 VL model:

```python
qwen_detector = QwenVLDetector(
    model_path="mlx-community/Qwen3-VL-8B-Instruct-4bit",
    adapter_path="output/v2-20260308-003328/checkpoint-861-mlx"
)
```

### 6.3 Memory Footprint

| Component | Memory |
|-----------|--------|
| YOLO11s | ~36 MB |
| Qwen 3 VL 8B (4-bit) | ~5 GB |
| LoRA adapter | ~333 MB |
| **Total peak** | **~6 GB** |
| Available on M4 16GB | ~10 GB headroom |

---

## 7. File Inventory

### 7.1 Core Application Files

| File | Purpose |
|------|---------|
| `demo_app.py` | Gradio web UI (YouTube + image upload) |
| `strawberry_detector.py` | Detection pipeline (YOLO + Qwen VL + RGB) |
| `strawberry_yolo_best.pt` | Fine-tuned YOLO11s weights (18.3 MB) |

### 7.2 Training & Evaluation Files

| File | Purpose |
|------|---------|
| `train_strawberry_yolo.py` | YOLO training script (Mac M4 MPS) |
| `qa_pipeline/train_runpod.sh` | Qwen fine-tuning script (RunPod) |
| `qa_pipeline/output/train.jsonl` | Training data (2,291 samples) |
| `qa_pipeline/output/val.jsonl` | Validation data (255 samples) |
| `qa_pipeline/crops/` | 18,256 cropped strawberry images |
| `compare_models.py` | Base vs fine-tuned comparison script |
| `convert_adapter_to_mlx.py` | PEFT -> MLX adapter converter |

### 7.3 Output Artifacts

| File / Directory | Purpose |
|-----------------|---------|
| `output/v2-20260308-003328/checkpoint-861/` | PEFT LoRA adapter (PyTorch) |
| `output/v2-20260308-003328/checkpoint-861-mlx/` | MLX LoRA adapter (for Apple Silicon) |
| `output/v2-20260308-003328/images/` | Training curves (loss, accuracy, LR) |
| `output/v2-20260308-003328/logging.jsonl` | Step-by-step training metrics |
| `comparison_results/` | 50-sample comparison results + charts |

---

## 8. Recommendations for Future Training

### 8.1 Dataset Improvements

| Area | Current | Recommendation |
|------|---------|----------------|
| **Training samples** | 2,291 | Increase to 5,000-10,000 for stronger generalization |
| **Validation set** | 255 (10%) | Maintain 10-15% split |
| **Question diversity** | ~5 question types | Add 10+ question templates to reduce overfitting to phrasing |
| **Edge cases** | Limited | Add more samples of: partially occluded berries, mixed-ripeness clusters, rare diseases |
| **Image diversity** | Single source | Include images from multiple greenhouses, lighting conditions, and camera angles |

### 8.2 Training Hyperparameters

| Parameter | Current | Suggestion | Rationale |
|-----------|---------|------------|-----------|
| **LoRA rank** | 32 | Try 16 and 64 | Rank 16 may suffice (fewer params, less overfitting). Rank 64 may capture more nuance |
| **Epochs** | 3 | Try 5 with early stopping | Eval loss was still decreasing at epoch 3; more epochs may help |
| **Learning rate** | 1e-4 | Try 5e-5 | Lower LR may produce smoother convergence and better final accuracy |
| **Gradient accumulation** | 8 | Keep or increase to 16 | Larger effective batch size stabilizes training |
| **LoRA alpha** | 64 (alpha/r = 2.0) | Try 32 (alpha/r = 1.0) | Lower scaling may prevent the adapter from dominating too aggressively |

### 8.3 Evaluation Improvements

| Area | Recommendation |
|------|----------------|
| **Sample size** | Increase evaluation from 50 to 255 (full validation set) for statistically significant results |
| **Cross-validation** | Use k-fold cross-validation to measure variance |
| **Human evaluation** | Conduct blind comparison of base vs fine-tuned outputs with domain experts |
| **Per-class breakdown** | Track accuracy separately for each ripeness stage (green, white, turning, ripe, overripe) |
| **Confidence calibration** | Measure whether model confidence correlates with actual correctness |

### 8.4 Architecture Improvements

| Area | Recommendation |
|------|----------------|
| **Base model upgrade** | Try Qwen 3 VL 32B (if GPU budget allows) for higher baseline capability |
| **Multi-task training** | Train with both classification labels AND free-text responses |
| **Vision encoder unfreezing** | Partially unfreeze ViT layers for strawberry-specific visual feature learning |
| **Adapter merging** | Once adapter is stable, merge into base weights for ~10% inference speedup |
| **Quantization-aware training** | Fine-tune with QLoRA (4-bit base + LoRA) to reduce memory and match deployment format |

---

## 9. Conclusion

Phase 2 successfully demonstrated that domain-specific fine-tuning significantly improves the Strawberry VLA system's analytical capabilities. The combination of a re-trained YOLO model (4,536 images) with a fine-tuned Qwen 3 VL model (2,291 Q&A samples, LoRA rank 32) delivers substantially better results than generic foundation models across all metrics.

The most dramatic improvement was in **harvest recommendation accuracy (+169%)**, proving that fine-tuning is essential for domain-specific decision-making tasks where the model needs to learn specific terminology and criteria.

The entire fine-tuned system runs locally on a Mac Mini M4 (16GB) using 4-bit quantization via MLX, with ~10 GB of memory headroom remaining. Total inference cost per strawberry analysis is approximately 10 seconds, making it practical for greenhouse operations.

---

*Report generated: March 2026*