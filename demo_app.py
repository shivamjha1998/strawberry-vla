"""
demo_app.py — Strawberry VLA Demo Interface (Task 1.4)

Interactive Gradio UI for the Strawberry Detection Pipeline.
Accepts YouTube URLs or image uploads, runs detection, and displays results.

Usage:
    python demo_app.py
    python demo_app.py --port 7861
    python demo_app.py --share   # Creates a public URL (for remote demo)

Then open http://localhost:7860 in your browser.
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

# Import our pipeline components
from strawberry_detector import (
    QwenVLDetector,
    YOLODetector,
    analyze_rgb_ripeness,
    visualize_detections,
    RIPENESS_COLORS,
)

# ─── Global State ────────────────────────────────────────────────
# Load model once and keep in memory for the demo session
qwen_detector = None
TEMP_DIR = tempfile.mkdtemp(prefix="strawberry_demo_")
os.makedirs(os.path.join(TEMP_DIR, "frames"), exist_ok=True)
os.makedirs(os.path.join(TEMP_DIR, "outputs"), exist_ok=True)


def get_detector():
    """Lazy-load the Qwen VL detector."""
    global qwen_detector
    if qwen_detector is None:
        qwen_detector = QwenVLDetector()
        qwen_detector.load()
    return qwen_detector


# ─── YouTube Processing ─────────────────────────────────────────

def download_youtube_video(url: str, max_duration: int = 120) -> str:
    """Download a YouTube video and return the file path."""
    output_path = os.path.join(TEMP_DIR, "video.mp4")

    # Remove old video if exists
    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = [
        "yt-dlp",
        "-f", "best[height<=720]",
        "--download-sections", f"*0-{max_duration}",
        "-o", output_path,
        "--no-playlist",
        "--quiet",
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr[:200]}")

    # Find the actual file (extension might differ)
    if os.path.exists(output_path):
        return output_path

    # Look for any video file
    for f in Path(TEMP_DIR).glob("video.*"):
        return str(f)

    raise RuntimeError("Video file not found after download")


def extract_sample_frames(video_path: str, num_frames: int = 8) -> list:
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        raise RuntimeError("Video has no frames")

    # Calculate frame indices (evenly spaced)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames_dir = os.path.join(TEMP_DIR, "frames")
    # Clean old frames
    for f in Path(frames_dir).glob("frame_*.jpg"):
        os.remove(f)

    frame_paths = []
    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = target_idx / fps if fps > 0 else 0

        # Resize if too large
        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        frame_path = os.path.join(frames_dir, f"frame_{len(frame_paths):04d}_t{timestamp:.1f}s.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(frame_path)

    cap.release()
    return frame_paths


# ─── Detection Logic ─────────────────────────────────────────────

def detect_single_image(image_path: str, do_disease: bool = False):
    """Run detection on a single image, return annotated image + results text."""
    detector = get_detector()

    # Detection
    t0 = time.time()
    detections = detector.detect_strawberries(image_path)
    det_time = time.time() - t0

    # RGB analysis
    img = cv2.imread(image_path)
    rgb_analyses = []
    if img is not None:
        for det in detections:
            bbox = det.get("bbox_2d", [])
            if len(bbox) == 4:
                rgb = analyze_rgb_ripeness(img, bbox)
                rgb_analyses.append(rgb)
            else:
                rgb_analyses.append({})

    # Disease
    disease_result = None
    if do_disease:
        disease_result = detector.assess_disease(image_path)

    # Create annotated image
    annotated_path = os.path.join(TEMP_DIR, "outputs", f"annotated_{Path(image_path).name}")
    if detections:
        visualize_detections(image_path, detections, annotated_path, rgb_analyses)
    else:
        # Copy original if no detections
        import shutil
        shutil.copy2(image_path, annotated_path)

    # Build results text
    results = build_results_text(detections, rgb_analyses, disease_result, det_time)

    # Build JSON data
    json_data = {
        "detections": detections,
        "rgb_analyses": rgb_analyses,
        "disease_assessment": disease_result,
        "detection_time_seconds": round(det_time, 2)
    }

    return annotated_path, results, json.dumps(json_data, indent=2)


def build_results_text(detections, rgb_analyses, disease_result, det_time):
    """Format detection results as readable text."""
    lines = []
    lines.append(f"## Detection Results")
    lines.append(f"**Strawberries found: {len(detections)}** (in {det_time:.1f}s)\n")

    if not detections:
        lines.append("No strawberries detected in this frame.")
        return "\n".join(lines)

    for i, det in enumerate(detections):
        ripeness = det.get("ripeness", "unknown")
        bbox = det.get("bbox_2d", [])
        size = det.get("size", "?")
        color_desc = det.get("color_description", "")
        conf = det.get("confidence", "?")

        emoji = {"green": "🟢", "white": "⚪", "turning": "🟠",
                 "nearly_ripe": "🟡", "ripe": "🔴", "overripe": "🟤"}.get(ripeness, "❓")

        lines.append(f"### {emoji} Strawberry #{i+1}")
        lines.append(f"- **Ripeness:** {ripeness} | **Size:** {size} | **Confidence:** {conf}")
        lines.append(f"- **Position:** {bbox}")

        if color_desc:
            lines.append(f"- **Color:** {color_desc}")

        # RGB analysis
        if i < len(rgb_analyses) and "error" not in rgb_analyses[i]:
            rgb = rgb_analyses[i]
            lines.append(f"- **RGB Analysis:**")
            lines.append(f"  - Mean RGB: ({rgb.get('mean_rgb', ['?','?','?'])})")
            lines.append(f"  - Red pixels: {rgb.get('red_pixel_pct', '?')}%")
            lines.append(f"  - Ripeness score: {rgb.get('ripeness_percentage', '?')}%")
            lines.append(f"  - Harvest ready: {'✅ Yes' if rgb.get('harvest_ready') else '❌ No'}")

        lines.append("")

    # Disease
    if disease_result:
        lines.append("### 🔬 Disease Assessment")
        health = disease_result.get("overall_plant_health", "unknown")
        health_emoji = {"healthy": "✅", "mild_issues": "⚠️",
                       "moderate_issues": "🟠", "severe_issues": "🔴"}.get(health, "❓")
        lines.append(f"**Overall health:** {health_emoji} {health}")

        diseases = disease_result.get("diseases_detected", [])
        if diseases:
            for d in diseases:
                lines.append(f"- {d.get('disease', '?')}: {d.get('severity', '?')} "
                           f"({d.get('location', '?')})")
        else:
            lines.append("No diseases detected.")

        notes = disease_result.get("notes", "")
        if notes:
            lines.append(f"\n*{notes}*")

    return "\n".join(lines)


# ─── Gradio Handlers ─────────────────────────────────────────────

def handle_youtube(url: str, num_frames: int, do_disease: bool, progress=gr.Progress()):
    """Process a YouTube URL end-to-end."""
    if not url or not url.strip():
        return [], "Please enter a YouTube URL.", "{}"

    try:
        # Step 1: Download
        progress(0.1, desc="Downloading video...")
        video_path = download_youtube_video(url.strip())

        # Step 2: Extract frames
        progress(0.2, desc="Extracting frames...")
        frame_paths = extract_sample_frames(video_path, num_frames=int(num_frames))

        if not frame_paths:
            return [], "No frames could be extracted.", "{}"

        # Step 3: Detect strawberries in each frame
        all_annotated = []
        all_results_text = []
        all_json = []

        for i, frame_path in enumerate(frame_paths):
            progress(0.2 + 0.7 * (i / len(frame_paths)),
                     desc=f"Analyzing frame {i+1}/{len(frame_paths)}...")

            annotated_path, results_text, json_str = detect_single_image(
                frame_path, do_disease=do_disease
            )
            all_annotated.append(annotated_path)
            all_results_text.append(f"---\n## Frame {i+1} (t={Path(frame_path).stem.split('_t')[-1]})\n{results_text}")
            all_json.append(json.loads(json_str))

        progress(1.0, desc="Done!")

        combined_text = "\n\n".join(all_results_text)

        # Summary stats
        total_strawberries = sum(len(j.get("detections", [])) for j in all_json)
        summary = (f"# Summary\n"
                   f"**Frames analyzed:** {len(frame_paths)} | "
                   f"**Total strawberries:** {total_strawberries} | "
                   f"**Avg per frame:** {total_strawberries/max(1,len(frame_paths)):.1f}\n\n")

        return (
            all_annotated,
            summary + combined_text,
            json.dumps({"frames": all_json}, indent=2)
        )

    except Exception as e:
        return [], f"Error: {str(e)}", "{}"


def handle_image_upload(image, do_disease: bool):
    """Process an uploaded image."""
    if image is None:
        return None, "Please upload an image.", "{}"

    try:
        # Save uploaded image
        upload_path = os.path.join(TEMP_DIR, "uploaded.jpg")
        if isinstance(image, np.ndarray):
            cv2.imwrite(upload_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif isinstance(image, str):
            upload_path = image
        else:
            return None, "Unsupported image format.", "{}"

        annotated_path, results_text, json_str = detect_single_image(
            upload_path, do_disease=do_disease
        )

        # Read annotated image back
        annotated = cv2.imread(annotated_path)
        if annotated is not None:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        return annotated, results_text, json_str

    except Exception as e:
        return None, f"Error: {str(e)}", "{}"


# ─── Build UI ────────────────────────────────────────────────────

def build_demo():
    """Build the Gradio interface."""

    with gr.Blocks(
        title="Strawberry VLA Demo",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 1em; }
        .results-box { max-height: 600px; overflow-y: auto; }
        """
    ) as demo:

        gr.Markdown("""
        # 🍓 Strawberry VLA — Detection Demo
        **Vision-Language-Action system for strawberry greenhouse operations**

        Detects strawberries, assesses ripeness (via VL + RGB analysis), and checks for diseases.
        Powered by Qwen 2.5 VL running locally on Mac Mini M4.
        """)

        with gr.Tabs():

            # ─── Tab 1: YouTube ───
            with gr.TabItem("📹 YouTube Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        yt_url = gr.Textbox(
                            label="YouTube URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                            info="Paste a strawberry greenhouse video URL"
                        )
                        with gr.Row():
                            yt_num_frames = gr.Slider(
                                minimum=2, maximum=16, value=6, step=1,
                                label="Frames to analyze",
                                info="More frames = more thorough but slower"
                            )
                            yt_disease = gr.Checkbox(
                                label="Disease detection",
                                value=False,
                                info="Slower but checks for powdery mildew, anthracnose"
                            )
                        yt_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")

                        gr.Markdown("""
                        **Sample videos to try:**
                        - Search YouTube for "strawberry greenhouse harvest"
                        - Search "いちご ハウス 収穫" for Japanese greenhouse footage
                        - Any close-up strawberry video works
                        """)

                    with gr.Column(scale=2):
                        yt_gallery = gr.Gallery(
                            label="Detected Frames",
                            columns=3,
                            height=400,
                            object_fit="contain"
                        )

                with gr.Row():
                    with gr.Column():
                        yt_results = gr.Markdown(
                            label="Detection Results",
                            value="*Results will appear here after analysis*"
                        )
                    with gr.Column():
                        yt_json = gr.Code(
                            label="Raw JSON Output",
                            language="json",
                            value="{}",
                            lines=15
                        )

                yt_btn.click(
                    fn=handle_youtube,
                    inputs=[yt_url, yt_num_frames, yt_disease],
                    outputs=[yt_gallery, yt_results, yt_json]
                )

            # ─── Tab 2: Image Upload ───
            with gr.TabItem("🖼️ Image Upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            label="Upload Strawberry Image",
                            type="numpy"
                        )
                        img_disease = gr.Checkbox(
                            label="Disease detection",
                            value=False
                        )
                        img_btn = gr.Button("🔍 Detect Strawberries", variant="primary")

                    with gr.Column(scale=1):
                        img_output = gr.Image(
                            label="Detection Result",
                            type="numpy"
                        )

                with gr.Row():
                    with gr.Column():
                        img_results = gr.Markdown(
                            label="Results",
                            value="*Upload an image and click Detect*"
                        )
                    with gr.Column():
                        img_json = gr.Code(
                            label="Raw JSON",
                            language="json",
                            value="{}",
                            lines=15
                        )

                img_btn.click(
                    fn=handle_image_upload,
                    inputs=[img_input, img_disease],
                    outputs=[img_output, img_results, img_json]
                )

            # ─── Tab 3: About ───
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## System Architecture

                ```
                YouTube URL / Image Upload
                        │
                        ▼
                ┌──────────────────┐
                │  Video Download  │  (yt-dlp, 720p)
                │  Frame Extract   │  (OpenCV)
                └──────────────────┘
                        │
                        ▼
                ┌──────────────────┐
                │  Qwen 2.5 VL     │  Strawberry detection
                │  (7B, 4-bit)     │  Bounding boxes (JSON)
                │                  │  Ripeness assessment
                │                  │  Disease detection
                └──────────────────┘
                        │
                        ▼
                ┌──────────────────┐
                │  RGB Analysis    │  Color-based ripeness
                │  (OpenCV)        │  Red/green pixel ratios
                └──────────────────┘
                        │
                        ▼
                  Annotated Results
                ```

                ## Capabilities

                | Feature | Method | Status |
                |---------|--------|--------|
                | Strawberry detection | Qwen 2.5 VL grounding | ✅ Working |
                | Ripeness assessment | VL + RGB analysis | ✅ Working |
                | Disease detection | Qwen 2.5 VL | ✅ Working |
                | 3D calibration | Multi-camera triangulation | 🔧 Phase 3 |
                | YOLO fine-tuned | Custom strawberry model | 🔧 Phase 2 |

                ## Hardware

                - **Inference:** Mac Mini M4, 16GB RAM
                - **Model:** Qwen 2.5 VL 7B (4-bit quantized via MLX)
                - **Fine-tuning target:** Tenstorrent AI Accelerator

                ## Ripeness Stages

                | Stage | Color | Description |
                |-------|-------|-------------|
                | 🟢 Green | Green | Unripe, not ready |
                | ⚪ White | Pale/white | Early development |
                | 🟠 Turning | Partial red | Getting close |
                | 🔴 Ripe | Full red | Ready to harvest |
                | 🟤 Overripe | Dark red/brown | Past peak |
                """)

        gr.Markdown("""
        ---
        *Strawberry VLA Demo — Running on Mac Mini M4 with Qwen 2.5 VL (mlx-vlm)*
        """)

    return demo


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Strawberry VLA Demo")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--preload", action="store_true",
                        help="Pre-load model before starting UI")
    args = parser.parse_args()

    print("=" * 60)
    print("🍓 Strawberry VLA — Demo Interface")
    print("=" * 60)
    print(f"  Temp dir: {TEMP_DIR}")
    print(f"  Port: {args.port}")
    print(f"  Share: {args.share}")

    if args.preload:
        print("\nPre-loading model...")
        get_detector()
        print("✓ Model ready")

    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
