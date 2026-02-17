"""
demo_app.py — Strawberry VLA Demo Interface (Updated for YOLO11)

Uses YOLO11 as primary detector (~10ms) + optional Qwen VL for detailed analysis.

Usage:
    python demo_app.py                     # Normal (YOLO fast mode)
    python demo_app.py --preload           # Pre-load models before UI
    python demo_app.py --share             # Public URL for remote demo
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from strawberry_detector import (
    YOLODetector,
    QwenVLDetector,
    analyze_rgb_ripeness,
    visualize_detections,
    RIPENESS_COLORS,
)

# ─── Global State ─────────────────────────────────────────────────────────────
yolo_detector = None
qwen_detector = None
TEMP_DIR = tempfile.mkdtemp(prefix="strawberry_demo_")
os.makedirs(os.path.join(TEMP_DIR, "frames"), exist_ok=True)
os.makedirs(os.path.join(TEMP_DIR, "outputs"), exist_ok=True)


def get_yolo():
    global yolo_detector
    if yolo_detector is None:
        yolo_detector = YOLODetector("strawberry_yolo_best.pt")
        yolo_detector.load()
    return yolo_detector


def get_qwen():
    global qwen_detector
    if qwen_detector is None:
        qwen_detector = QwenVLDetector()
        qwen_detector.load()
    return qwen_detector


# ─── YouTube Processing ──────────────────────────────────────────────────────

def download_youtube_video(url: str, max_duration: int = 120) -> str:
    output_path = os.path.join(TEMP_DIR, "video.mp4")
    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = [
        "yt-dlp", "-f", "best[height<=720]",
        "--download-sections", f"*0-{max_duration}",
        "-o", output_path, "--no-playlist", "--quiet", url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr[:200]}")

    if os.path.exists(output_path):
        return output_path
    for f in Path(TEMP_DIR).glob("video.*"):
        return str(f)
    raise RuntimeError("Video file not found after download")


def extract_sample_frames(video_path: str, num_frames: int = 8) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0:
        raise RuntimeError("Video has no frames")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames_dir = os.path.join(TEMP_DIR, "frames")
    for f in Path(frames_dir).glob("frame_*.jpg"):
        os.remove(f)

    frame_paths = []
    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = target_idx / fps if fps > 0 else 0
        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frame_path = os.path.join(frames_dir, f"frame_{len(frame_paths):04d}_t{timestamp:.1f}s.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(frame_path)

    cap.release()
    return frame_paths


# ─── Detection Logic ──────────────────────────────────────────────────────────

def detect_single_image(image_path: str, do_disease: bool = False, do_detailed: bool = False):
    """Run YOLO11 detection + RGB + optional Qwen VL analysis."""
    yolo = get_yolo()

    # YOLO detection (fast)
    t0 = time.time()
    detections = yolo.detect(image_path)
    det_time = time.time() - t0

    # RGB analysis
    img = cv2.imread(image_path)
    rgb_analyses = []
    if img is not None:
        for det in detections:
            bbox = det.get("bbox_2d", [])
            rgb_analyses.append(
                analyze_rgb_ripeness(img, bbox) if len(bbox) == 4 else {}
            )

    # Qwen VL per-crop analysis (optional)
    qwen_analyses = []
    if do_detailed and detections:
        qwen = get_qwen()
        if img is not None:
            h, w = img.shape[:2]
            for det in detections:
                bbox = det.get("bbox_2d", [])
                if len(bbox) != 4:
                    qwen_analyses.append({})
                    continue
                x1, y1, x2, y2 = bbox
                pad_x, pad_y = int((x2-x1)*0.1), int((y2-y1)*0.1)
                crop = img[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
                if crop.size == 0:
                    qwen_analyses.append({})
                    continue
                crop_path = os.path.join(TEMP_DIR, "outputs", f"crop_{id(det)}.jpg")
                cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                try:
                    qwen_analyses.append(qwen.assess_ripeness_detail(crop_path))
                finally:
                    if os.path.exists(crop_path):
                        os.unlink(crop_path)

    # Disease (optional)
    disease_result = None
    if do_disease:
        qwen = get_qwen()
        disease_result = qwen.assess_disease(image_path)

    # Annotated image
    annotated_path = os.path.join(TEMP_DIR, "outputs", f"annotated_{Path(image_path).name}")
    if detections:
        visualize_detections(image_path, detections, annotated_path, rgb_analyses)
    else:
        import shutil
        shutil.copy2(image_path, annotated_path)

    results_text = build_results_text(detections, rgb_analyses, qwen_analyses, disease_result, det_time)
    json_data = {
        "detections": detections, "rgb_analyses": rgb_analyses,
        "qwen_analyses": qwen_analyses, "disease_assessment": disease_result,
        "detection_time_ms": round(det_time * 1000, 1)
    }

    return annotated_path, results_text, json.dumps(json_data, indent=2)


def build_results_text(detections, rgb_analyses, qwen_analyses, disease_result, det_time):
    lines = []
    lines.append(f"## Detection Results")
    lines.append(f"**Strawberries found: {len(detections)}** (YOLO11: {det_time*1000:.1f}ms)\n")

    if not detections:
        lines.append("No strawberries detected in this frame.")
        return "\n".join(lines)

    for i, det in enumerate(detections):
        ripeness = det.get("ripeness", "unknown")
        yolo_cls = det.get("yolo_class", ripeness)
        conf = det.get("confidence_score", 0)
        emoji = {"green": "🟢", "white": "⚪", "turning": "🟠",
                 "ripe": "🔴", "overripe": "🟤", "flower": "🌸"}.get(ripeness, "❓")

        lines.append(f"### {emoji} Strawberry #{i+1}")
        lines.append(f"- **Class:** {yolo_cls} | **Confidence:** {conf:.0%}")
        lines.append(f"- **Position:** {det.get('bbox_2d')}")

        if i < len(rgb_analyses) and "error" not in rgb_analyses[i]:
            rgb = rgb_analyses[i]
            lines.append(f"- **RGB Analysis:** ripe={rgb.get('ripeness_percentage', '?')}% | "
                         f"red={rgb.get('red_pixel_pct', '?')}% | "
                         f"harvest={'✅' if rgb.get('harvest_ready') else '❌'}")

        if i < len(qwen_analyses) and qwen_analyses[i] and "error" not in qwen_analyses[i]:
            qa = qwen_analyses[i]
            lines.append(f"- **Qwen VL:** {qa.get('ripeness_stage', '?')} "
                         f"({qa.get('ripeness_percentage', '?')}%) — {qa.get('notes', '')}")

        lines.append("")

    if disease_result:
        lines.append("### 🔬 Disease Assessment")
        health = disease_result.get("overall_plant_health", "unknown")
        h_emoji = {"healthy": "✅", "mild_issues": "⚠️", "moderate_issues": "🟠", "severe_issues": "🔴"}.get(health, "❓")
        lines.append(f"**Health:** {h_emoji} {health}")
        for d in disease_result.get("diseases_detected", []):
            lines.append(f"- {d.get('disease')}: {d.get('severity')} ({d.get('location')})")
        if not disease_result.get("diseases_detected"):
            lines.append("No diseases detected.")

    return "\n".join(lines)


# ─── Gradio Handlers ─────────────────────────────────────────────────────────

def handle_youtube(url, num_frames, do_disease, do_detailed, progress=gr.Progress()):
    if not url or not url.strip():
        return [], "Please enter a YouTube URL.", "{}"
    try:
        progress(0.1, desc="Downloading video...")
        video_path = download_youtube_video(url.strip())
        progress(0.2, desc="Extracting frames...")
        frame_paths = extract_sample_frames(video_path, num_frames=int(num_frames))
        if not frame_paths:
            return [], "No frames extracted.", "{}"

        all_annotated, all_text, all_json = [], [], []
        for i, fp in enumerate(frame_paths):
            progress(0.2 + 0.7 * (i / len(frame_paths)),
                     desc=f"Analyzing frame {i+1}/{len(frame_paths)}...")
            ann_path, text, js = detect_single_image(fp, do_disease=do_disease, do_detailed=do_detailed)
            all_annotated.append(ann_path)
            all_text.append(f"---\n## Frame {i+1} (t={Path(fp).stem.split('_t')[-1]})\n{text}")
            all_json.append(json.loads(js))

        progress(1.0, desc="Done!")
        total = sum(len(j.get("detections", [])) for j in all_json)
        summary = (f"# Summary\n**Frames:** {len(frame_paths)} | "
                   f"**Strawberries:** {total} | "
                   f"**Avg/frame:** {total/max(1,len(frame_paths)):.1f}\n\n")
        return all_annotated, summary + "\n\n".join(all_text), json.dumps({"frames": all_json}, indent=2)
    except Exception as e:
        return [], f"Error: {str(e)}", "{}"


def handle_image_upload(image, do_disease, do_detailed):
    if image is None:
        return None, "Please upload an image.", "{}"
    try:
        upload_path = os.path.join(TEMP_DIR, "uploaded.jpg")
        if isinstance(image, np.ndarray):
            cv2.imwrite(upload_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif isinstance(image, str):
            upload_path = image
        else:
            return None, "Unsupported format.", "{}"

        ann_path, text, js = detect_single_image(upload_path, do_disease=do_disease, do_detailed=do_detailed)
        annotated = cv2.imread(ann_path)
        if annotated is not None:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return annotated, text, js
    except Exception as e:
        return None, f"Error: {str(e)}", "{}"


# ─── Build UI ─────────────────────────────────────────────────────────────────

def build_demo():
    with gr.Blocks(title="Strawberry VLA Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🍓 Strawberry VLA — Detection Demo
        **YOLO11 trained detector** (~10ms/frame) + Qwen 2.5 VL analysis + RGB ripeness scoring.
        Running locally on Mac Mini M4.
        """)

        with gr.Tabs():
            # Tab 1: YouTube
            with gr.TabItem("📹 YouTube Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        yt_url = gr.Textbox(label="YouTube URL",
                                            placeholder="https://www.youtube.com/watch?v=...")
                        with gr.Row():
                            yt_frames = gr.Slider(2, 16, 6, step=1, label="Frames")
                            yt_disease = gr.Checkbox(label="Disease detection", value=False)
                            yt_detailed = gr.Checkbox(label="Qwen VL detailed", value=False)
                        yt_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        yt_gallery = gr.Gallery(label="Detected Frames", columns=3, height=400)
                with gr.Row():
                    yt_results = gr.Markdown(value="*Results appear after analysis*")
                    yt_json = gr.Code(label="JSON", language="json", value="{}", lines=15)
                yt_btn.click(handle_youtube, [yt_url, yt_frames, yt_disease, yt_detailed],
                             [yt_gallery, yt_results, yt_json])

            # Tab 2: Image Upload
            with gr.TabItem("🖼️ Image Upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(label="Upload Strawberry Image", type="numpy")
                        img_disease = gr.Checkbox(label="Disease detection", value=False)
                        img_detailed = gr.Checkbox(label="Qwen VL detailed", value=False)
                        img_btn = gr.Button("🔍 Detect", variant="primary")
                    with gr.Column(scale=1):
                        img_output = gr.Image(label="Result", type="numpy")
                with gr.Row():
                    img_results = gr.Markdown(value="*Upload an image and click Detect*")
                    img_json = gr.Code(label="JSON", language="json", value="{}", lines=15)
                img_btn.click(handle_image_upload, [img_input, img_disease, img_detailed],
                              [img_output, img_results, img_json])

            # Tab 3: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## Architecture (Updated)

                ```
                Frame → YOLO11 (<10ms) → bounding boxes + ripeness class
                            ↓
                     RGB Analysis (instant) → color-based ripeness %
                            ↓ (optional)
                     Qwen 2.5 VL (10-30s/crop) → detailed analysis + disease
                ```

                ## Performance (Mac Mini M4)

                | Mode | Speed | What you get |
                |------|-------|-------------|
                | **Fast** (default) | ~10ms/frame | YOLO boxes + RGB ripeness |
                | **Detailed** | ~10-30s/crop | + Qwen VL analysis per strawberry |
                | **Disease** | ~10-30s/frame | + Disease assessment |

                ## Model Info
                - **YOLO11n**: Trained on 1060 images, mAP@50=86.1%, ~6MB
                - **Qwen 2.5 VL 7B**: 4-bit quantized via MLX, ~4.5GB
                """)

        gr.Markdown("---\n*Strawberry VLA Demo — YOLO11 + Qwen 2.5 VL on Mac Mini M4*")
    return demo


def main():
    parser = argparse.ArgumentParser(description="Strawberry VLA Demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("🍓 Strawberry VLA — Demo Interface (YOLO11)")
    print("=" * 60)

    if args.preload:
        print("\nPre-loading YOLO11...")
        get_yolo()
        print("✓ YOLO11 ready")
        print("\nPre-loading Qwen 2.5 VL...")
        get_qwen()
        print("✓ Qwen 2.5 VL ready")

    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()