"""
demo_app.py — Strawberry VLA Demo Interface

Uses YOLO11 as primary detector (~10ms) + optional Qwen VL for detailed analysis.
Pastel UI with full English/Japanese language switching. Translations in locales/*.json.

Usage:
    python demo_app.py                     # Normal (YOLO fast mode)
    python demo_app.py --preload           # Pre-load models before UI
    python demo_app.py --share             # Public URL for remote demo
"""

import argparse
import json
import os
import shutil
import subprocess
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


# ─── Locale Loading ───────────────────────────────────────────────────────────

LOCALES_DIR = Path(__file__).parent / "locales"
_locales = {}


def load_locales():
    global _locales
    for lang_file in LOCALES_DIR.glob("*.json"):
        with open(lang_file, encoding="utf-8") as f:
            _locales[lang_file.stem] = json.load(f)


def t(key, lang="en"):
    return _locales.get(lang, _locales.get("en", {})).get(key, key)


load_locales()


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
        frame_path = os.path.join(
            frames_dir, f"frame_{len(frame_paths):04d}_t{timestamp:.1f}s.jpg"
        )
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(frame_path)

    cap.release()
    return frame_paths


# ─── Detection Logic ──────────────────────────────────────────────────────────

def detect_single_image(image_path, do_disease=False, do_detailed=False):
    yolo = get_yolo()

    t0 = time.time()
    detections = yolo.detect(image_path)
    det_time = time.time() - t0

    img = cv2.imread(image_path)
    rgb_analyses = []
    if img is not None:
        for det in detections:
            bbox = det.get("bbox_2d", [])
            rgb_analyses.append(
                analyze_rgb_ripeness(img, bbox) if len(bbox) == 4 else {}
            )

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
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                crop = img[
                    max(0, y1 - pad_y):min(h, y2 + pad_y),
                    max(0, x1 - pad_x):min(w, x2 + pad_x),
                ]
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

    disease_result = None
    if do_disease:
        qwen = get_qwen()
        disease_result = qwen.assess_disease(image_path)

    annotated_path = os.path.join(
        TEMP_DIR, "outputs", f"annotated_{Path(image_path).name}"
    )
    if detections:
        visualize_detections(image_path, detections, annotated_path, rgb_analyses)
    else:
        shutil.copy2(image_path, annotated_path)

    return annotated_path, detections, rgb_analyses, qwen_analyses, disease_result, det_time


def build_results_text(detections, rgb_analyses, qwen_analyses, disease_result, det_time, lang="en"):
    lines = []
    lines.append(f"### {t('detection_results', lang)}")
    lines.append(
        f"**{t('strawberries_found', lang)}: {len(detections)}** — "
        f"YOLO11: {det_time * 1000:.1f}ms\n"
    )

    if not detections:
        lines.append(t("no_strawberries", lang))
        return "\n".join(lines)

    for i, det in enumerate(detections):
        yolo_cls = det.get("yolo_class", det.get("ripeness", "unknown"))
        conf = det.get("confidence_score", 0)

        lines.append(f"**{t('strawberry_n', lang)} #{i + 1}**")
        lines.append(
            f"- {t('class_label', lang)}: {yolo_cls} | "
            f"{t('confidence_label', lang)}: {conf:.0%}"
        )
        lines.append(f"- {t('position_label', lang)}: {det.get('bbox_2d')}")

        if i < len(rgb_analyses) and "error" not in rgb_analyses[i]:
            rgb = rgb_analyses[i]
            harvest = (
                t("harvest_yes", lang) if rgb.get("harvest_ready")
                else t("harvest_no", lang)
            )
            lines.append(
                f"- {t('rgb_label', lang)}: "
                f"{t('ripe_pct', lang)}={rgb.get('ripeness_percentage', '?')}% | "
                f"{t('red_pct', lang)}={rgb.get('red_pixel_pct', '?')}% | "
                f"{t('harvest_ready', lang)}: {harvest}"
            )

        if i < len(qwen_analyses) and qwen_analyses[i] and "error" not in qwen_analyses[i]:
            qa = qwen_analyses[i]
            lines.append(
                f"- {t('qwen_label', lang)}: {qa.get('ripeness_stage', '?')} "
                f"({qa.get('ripeness_percentage', '?')}%) — {qa.get('notes', '')}"
            )
        lines.append("")

    if disease_result:
        lines.append(f"**{t('disease_title', lang)}**")
        health = disease_result.get("overall_plant_health", "unknown")
        lines.append(f"{t('health_label', lang)}: {health}")
        for d in disease_result.get("diseases_detected", []):
            lines.append(f"- {d.get('disease')}: {d.get('severity')} ({d.get('location')})")
        if not disease_result.get("diseases_detected"):
            lines.append(t("no_disease", lang))

    return "\n".join(lines)


# ─── About section builder ───────────────────────────────────────────────────

def build_about_md(lang="en"):
    return (
        f"### {t('about_architecture', lang)}\n\n"
        f"{t('about_arch_text', lang)}\n\n"
        f"---\n\n"
        f"### {t('about_performance', lang)}\n\n"
        f"| {t('perf_mode_header', lang)} | {t('perf_speed_header', lang)} | {t('perf_output_header', lang)} |\n"
        f"|------|-------|--------|\n"
        f"| {t('perf_fast', lang)} | {t('perf_fast_speed', lang)} | {t('perf_fast_desc', lang)} |\n"
        f"| {t('perf_detailed', lang)} | {t('perf_detailed_speed', lang)} | {t('perf_detailed_desc', lang)} |\n"
        f"| {t('perf_disease', lang)} | {t('perf_disease_speed', lang)} | {t('perf_disease_desc', lang)} |\n\n"
        f"---\n\n"
        f"### {t('about_models', lang)}\n\n"
        f"- {t('model_yolo', lang)}\n"
        f"- {t('model_qwen', lang)}"
    )


# ─── Gradio Handlers ─────────────────────────────────────────────────────────

def handle_youtube(url, num_frames, do_disease, do_detailed, lang, progress=gr.Progress()):
    if not url or not url.strip():
        return [], t("enter_url", lang), "{}"
    try:
        progress(0.1, desc="Downloading...")
        video_path = download_youtube_video(url.strip())
        progress(0.2, desc="Extracting frames...")
        frame_paths = extract_sample_frames(video_path, num_frames=int(num_frames))
        if not frame_paths:
            return [], t("no_frames", lang), "{}"

        all_annotated, all_text, all_json = [], [], []
        for i, fp in enumerate(frame_paths):
            progress(
                0.2 + 0.7 * (i / len(frame_paths)),
                desc=f"Frame {i + 1}/{len(frame_paths)}...",
            )
            ann_path, dets, rgb, qwen, disease, dt = detect_single_image(
                fp, do_disease=do_disease, do_detailed=do_detailed
            )
            all_annotated.append(ann_path)
            text = build_results_text(dets, rgb, qwen, disease, dt, lang)
            all_text.append(
                f"---\n**Frame {i + 1}** "
                f"(t={Path(fp).stem.split('_t')[-1]})\n\n{text}"
            )
            all_json.append({
                "detections": dets, "rgb_analyses": rgb,
                "qwen_analyses": qwen, "disease_assessment": disease,
                "detection_time_ms": round(dt * 1000, 1),
            })

        progress(1.0, desc="Done")
        total = sum(len(j.get("detections", [])) for j in all_json)
        summary = (
            f"### {t('detection_results', lang)}\n"
            f"**{t('summary_frames', lang)}:** {len(frame_paths)} | "
            f"**{t('summary_total', lang)}:** {total} | "
            f"**{t('summary_avg', lang)}:** {total / max(1, len(frame_paths)):.1f}\n\n"
        )
        return (
            all_annotated,
            summary + "\n\n".join(all_text),
            json.dumps({"frames": all_json}, indent=2),
        )
    except Exception as e:
        return [], f"{t('error_prefix', lang)}: {str(e)}", "{}"


def handle_image_upload(image, do_disease, do_detailed, lang):
    if image is None:
        return None, t("upload_placeholder", lang), "{}"
    try:
        upload_path = os.path.join(TEMP_DIR, "uploaded.jpg")
        if isinstance(image, np.ndarray):
            cv2.imwrite(
                upload_path,
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
        elif isinstance(image, str):
            upload_path = image
        else:
            return None, f"{t('error_prefix', lang)}: Unsupported format.", "{}"

        ann_path, dets, rgb, qwen, disease, dt = detect_single_image(
            upload_path, do_disease=do_disease, do_detailed=do_detailed
        )
        text = build_results_text(dets, rgb, qwen, disease, dt, lang)
        json_data = {
            "detections": dets, "rgb_analyses": rgb,
            "qwen_analyses": qwen, "disease_assessment": disease,
            "detection_time_ms": round(dt * 1000, 1),
        }
        annotated = cv2.imread(ann_path)
        if annotated is not None:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return annotated, text, json.dumps(json_data, indent=2)
    except Exception as e:
        return None, f"{t('error_prefix', lang)}: {str(e)}", "{}"


# ─── Tab helpers (custom buttons instead of gr.Tabs for dynamic labels) ──────

def _select_tab(idx):
    """Return visibility + variant updates for 3 panels + 3 buttons."""
    panels = [gr.update(visible=(i == idx)) for i in range(3)]
    buttons = [gr.update(variant="primary" if i == idx else "secondary") for i in range(3)]
    return panels + buttons


def select_tab_0():
    return _select_tab(0)


def select_tab_1():
    return _select_tab(1)


def select_tab_2():
    return _select_tab(2)


# ─── Language Switcher ────────────────────────────────────────────────────────

def switch_language(lang):
    """Update every visible label / text when language changes."""
    return [
        # 0  title
        gr.update(value=f"# {t('title', lang)}\n\n{t('subtitle', lang)}"),
        # 1  description
        gr.update(value=t("description", lang)),
        # 2  yt_url
        gr.update(label=t("yt_url_label", lang), placeholder=t("yt_url_placeholder", lang)),
        # 3  yt_frames
        gr.update(label=t("frames_label", lang)),
        # 4  yt_disease
        gr.update(label=t("disease_label", lang)),
        # 5  yt_detailed
        gr.update(label=t("detailed_label", lang)),
        # 6  yt_btn
        gr.update(value=t("analyze_btn", lang)),
        # 7  yt_gallery
        gr.update(label=t("gallery_label", lang)),
        # 8  yt_results
        gr.update(value=f"*{t('results_placeholder', lang)}*"),
        # 9  img_input
        gr.update(label=t("upload_label", lang)),
        # 10 img_disease
        gr.update(label=t("disease_label", lang)),
        # 11 img_detailed
        gr.update(label=t("detailed_label", lang)),
        # 12 img_btn
        gr.update(value=t("detect_btn", lang)),
        # 13 img_output
        gr.update(label=t("result_image_label", lang)),
        # 14 img_results
        gr.update(value=f"*{t('upload_placeholder', lang)}*"),
        # 15 about_md
        gr.update(value=build_about_md(lang)),
        # 16 footer_md
        gr.update(value=t("footer", lang)),
        # 17-19  tab buttons
        gr.update(value=t("tab_video", lang)),
        gr.update(value=t("tab_image", lang)),
        gr.update(value=t("tab_about", lang)),
    ]


# ─── Pastel CSS ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Gradio CSS variable overrides ── */
:root, .gradio-container, .dark {
    --body-text-color: #1a1a1a !important;
    --block-label-text-color: #2a2a2a !important;
    --block-title-text-color: #2a2a2a !important;
    --block-label-background-fill: #edeae7 !important;
    --block-background-fill: transparent !important;
    --block-border-color: #ddd9d5 !important;
    --background-fill-primary: #f5f3f0 !important;
    --background-fill-secondary: #f5f3f0 !important;
    --input-background-fill: #ffffff !important;
    --input-placeholder-color: #888 !important;
    --checkbox-background-color: #ffffff !important;
    --checkbox-background-color-selected: #d4918e !important;
    --checkbox-border-color: #c8c3be !important;
    --checkbox-border-color-selected: #d4918e !important;
    --checkbox-label-background-fill: transparent !important;
    --checkbox-label-text-color: #2a2a2a !important;
    --panel-background-fill: transparent !important;
    --table-even-background-fill: transparent !important;
    --table-odd-background-fill: transparent !important;
    --slider-color: #d4918e !important;
    --neutral-50: #f5f3f0 !important;
    --neutral-100: #edeae7 !important;
    --neutral-200: #ddd9d5 !important;
    --neutral-300: #c8c3be !important;
    --neutral-400: #999 !important;
    --neutral-500: #777 !important;
    --neutral-600: #555 !important;
    --neutral-700: #3a3a3a !important;
    --neutral-800: #2a2a2a !important;
    --neutral-900: #1a1a1a !important;
    --neutral-950: #0a0a0a !important;
    --color-accent: #d4918e !important;
}

/* ── Base — full-width landscape ── */
.gradio-container {
    background: #f5f3f0 !important;
    color: #1a1a1a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", "Segoe UI", Arial, sans-serif !important;
    max-width: 100% !important;
    padding: 0 48px !important;
}

/* ── Header ── */
.gradio-container { position: relative; }
#lang-row {
    position: absolute !important;
    top: 16px !important;
    right: 48px !important;
    z-index: 10 !important;
    width: auto !important;
    padding: 0 !important;
    margin: 0 !important;
}
#title-text { text-align: center; margin-bottom: 0 !important; padding-top: 12px !important; }
#title-text h1 {
    font-size: 26px !important;
    font-weight: 600 !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    color: #1a1a1a !important;
    margin-bottom: 0 !important;
}
#title-text p {
    font-size: 15px !important;
    font-weight: 400 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #555 !important;
    margin-top: 2px !important;
}
#description-text p {
    color: #444 !important;
    font-size: 12px !important;
    text-align: center !important;
    letter-spacing: 0.3px !important;
    max-width: 700px;
    margin: 4px auto 0 !important;
    line-height: 1.5 !important;
}

/* ── Language selector ── */
#lang-selector { max-width: 160px; margin-left: auto; }
#lang-selector .wrap {
    background: #d4908f !important;
    border: 1px solid #d6d1cc !important;
    border-radius: 8px !important;
}
#lang-selector label {
    color: #fff !important;
    font-size: 12px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* ── Custom tab navigation ── */
#tab-nav {
    border-bottom: 1px solid #ddd9d5 !important;
    background: transparent !important;
    justify-content: center !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 16px !important;
}
#tab-nav button {
    border-radius: 0 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 12px 32px !important;
    transition: all 0.2s ease !important;
    min-width: auto !important;
}
/* Active tab (primary variant) */
#tab-nav button.primary,
#tab-nav .primary {
    background: transparent !important;
    color: #c47f7b !important;
    border: none !important;
    border-bottom: 2px solid #c47f7b !important;
    box-shadow: none !important;
}
#tab-nav button.primary:hover {
    background: transparent !important;
    transform: none !important;
    box-shadow: none !important;
}
/* Inactive tab (secondary variant) */
#tab-nav button.secondary,
#tab-nav .secondary {
    background: transparent !important;
    color: #777 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    box-shadow: none !important;
}
#tab-nav button.secondary:hover {
    background: transparent !important;
    color: #555 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Panels ── */
.panel, .form, .block {
    background: transparent !important;
    border: none !important;
}

/* ── Component label bars (Upload image, Detection Results, etc.) ── */
.gradio-container span.svelte-1gfkn6j,
.image-container .label-wrap,
.upload-container .label-wrap,
.gallery-container .label-wrap,
span[data-testid="block-label"] {
    background: #edeae7 !important;
    color: #3a3a3a !important;
    font-size: 13px !important;
}

/* ── Inputs ── */
.gr-input, .gr-text-input, textarea, input[type="text"] {
    background: #ffffff !important;
    border: 1px solid #d6d1cc !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s ease !important;
}
.gr-input:focus, .gr-text-input:focus, textarea:focus, input[type="text"]:focus {
    border-color: #d4918e !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(212, 145, 142, 0.12) !important;
}
input::placeholder, textarea::placeholder { color: #888 !important; }

/* ── Labels (brute-force every Gradio label variant) ── */
label, .gr-input-label, .label-wrap span,
.gradio-container label,
.gradio-container label span,
.gradio-container .label-text,
.gradio-container .group-text,
.gradio-container span[data-testid="block-info"],
.gradio-container .block label span,
.gradio-container .wrap > label,
.gradio-container [class*="label"] {
    color: #2a2a2a !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── Checkbox container ── */
.gr-checkbox-container, .checkbox-container,
input[type="checkbox"] ~ label,
.gr-check-radio ~ label {
    color: #3a3a3a !important;
    font-size: 13px !important;
    background: transparent !important;
}
.gr-form, .gr-group, .gr-block {
    background: transparent !important;
}

/* ── Action buttons (Analyze / Detect) ── */
.action-btn button,
.action-btn .primary {
    background: #d4918e !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 11px 32px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(212, 145, 142, 0.25) !important;
}
.action-btn button:hover,
.action-btn .primary:hover {
    background: #c47f7b !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(212, 145, 142, 0.35) !important;
}

/* ── Slider ── */
input[type="range"] { accent-color: #d4918e !important; }

/* ── Checkbox ── */
.gr-check-radio {
    background: #fff !important;
    border: 1.5px solid #c8c3be !important;
    border-radius: 5px !important;
}
input[type="checkbox"]:checked + .gr-check-radio,
.gr-checkbox-container input:checked ~ span {
    background: #d4918e !important;
    border-color: #d4918e !important;
}

/* ── Gallery ── */
.gallery-container, .grid-wrap {
    background: #ffffff !important;
    border: 1px solid #ddd9d5 !important;
    border-radius: 14px !important;
    overflow: hidden;
}
.gallery-item, .thumbnail-item {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid #e8e4e0 !important;
    transition: border-color 0.2s ease !important;
}
.gallery-item:hover, .thumbnail-item:hover {
    border-color: #d4918e !important;
}

/* ── Image component ── */
.image-container {
    background: #fff !important;
    border: 1px solid #ddd9d5 !important;
    border-radius: 14px !important;
    overflow: hidden;
}

/* ── Markdown output ── */
.prose, .markdown-text, .md {
    color: #2a2a2a !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}
.prose h3, .markdown-text h3, .md h3 {
    color: #1a1a1a !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-top: 18px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #ddd9d5 !important;
}
.prose strong, .markdown-text strong, .md strong {
    color: #1a1a1a !important;
    font-weight: 600 !important;
}

/* ── Accordion ── */
.accordion { border: 1px solid #ddd9d5 !important; border-radius: 10px !important; }
.accordion .label-wrap {
    background: #edeae7 !important;
    border-radius: 10px !important;
}

/* ── Code / JSON ── */
.code-wrap, .cm-editor {
    background: #faf9f7 !important;
    border: 1px solid #ddd9d5 !important;
    border-radius: 10px !important;
    font-size: 12px !important;
}
.cm-gutters {
    background: #edeae7 !important;
    border-right: 1px solid #ddd9d5 !important;
}
.cm-content, .cm-line { color: #444 !important; }

/* ── Footer ── */
#footer-text {
    text-align: center;
    padding-top: 16px;
    border-top: 1px solid #ddd9d5;
    margin-top: 24px;
}
#footer-text p {
    color: #888 !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* ── About tab ── */
#about-section .prose p { color: #333 !important; }

/* ── Table styling ── */
.prose table, .md table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 12px 0 !important;
}
.prose th, .md th {
    background: #edeae7 !important;
    color: #3a3a3a !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid #d6d1cc !important;
    text-align: left !important;
}
.prose td, .md td {
    padding: 8px 12px !important;
    border-bottom: 1px solid #ddd9d5 !important;
    color: #2a2a2a !important;
    font-size: 14px !important;
}

/* ── Misc ── */
.gr-padded { padding: 16px !important; }
footer { display: none !important; }
.built-with { display: none !important; }
.show-api { display: none !important; }
.wrap.default { background: transparent !important; border: none !important; }

/* ── Force all text to be dark ── */
.gradio-container * {
    --block-label-text-color: #3a3a3a !important;
    --block-label-background-fill: #edeae7 !important;
    --checkbox-label-background-fill: transparent !important;
    --checkbox-background-color: #ffffff !important;
    --input-background-fill: #ffffff !important;
}
/* Override any remaining Gradio dark backgrounds on component wrappers */
.gradio-container .wrap,
.gradio-container .container {
    background: transparent !important;
}
.gradio-container [class*="block"] > .wrap:first-child {
    background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f5f3f0; }
::-webkit-scrollbar-thumb { background: #c8c3be; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #999; }
"""


# ─── Build UI ─────────────────────────────────────────────────────────────────

def build_demo():
    with gr.Blocks(
        title="Strawberry VLA",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.red,
            secondary_hue=gr.themes.colors.stone,
            neutral_hue=gr.themes.colors.stone,
        ).set(
            body_background_fill="#f5f3f0",
            body_text_color="#1a1a1a",
            block_background_fill="transparent",
            block_label_background_fill="#edeae7",
            block_label_text_color="#2a2a2a",
            block_title_text_color="#2a2a2a",
            block_border_color="#ddd9d5",
            input_background_fill="#ffffff",
            input_placeholder_color="#888",
            checkbox_background_color="#ffffff",
            checkbox_background_color_selected="#d4918e",
            checkbox_border_color="#c8c3be",
            checkbox_border_color_selected="#d4918e",
            checkbox_label_background_fill="transparent",
            checkbox_label_text_color="#2a2a2a",
            panel_background_fill="transparent",
            slider_color="#d4918e",
        ),
    ) as demo:

        # ── Header ──
        with gr.Row(elem_id="lang-row"):
            lang = gr.Dropdown(
                choices=[("English", "en"), ("日本語", "ja")],
                value="en",
                label=t("language_label"),
                elem_id="lang-selector",
                interactive=True,
            )
        title_md = gr.Markdown(
            f"# {t('title')}\n\n{t('subtitle')}",
            elem_id="title-text",
        )
        desc_md = gr.Markdown(t("description"), elem_id="description-text")

        # ── Tab navigation (custom buttons — labels update on language change) ──
        with gr.Row(elem_id="tab-nav"):
            tab_btn_video = gr.Button(
                t("tab_video"), variant="primary", size="sm",
            )
            tab_btn_image = gr.Button(
                t("tab_image"), variant="secondary", size="sm",
            )
            tab_btn_about = gr.Button(
                t("tab_about"), variant="secondary", size="sm",
            )

        # ── Panel 1: YouTube Video ──
        with gr.Column(visible=True) as panel_video:
            with gr.Row():
                with gr.Column(scale=1):
                    yt_url = gr.Textbox(
                        label=t("yt_url_label"),
                        placeholder=t("yt_url_placeholder"),
                        elem_id="yt-url",
                    )
                    yt_frames = gr.Slider(
                        2, 16, 6, step=1, label=t("frames_label"),
                        elem_id="yt-frames",
                    )
                    with gr.Row():
                        yt_disease = gr.Checkbox(
                            label=t("disease_label"), value=False,
                            elem_id="yt-disease",
                        )
                        yt_detailed = gr.Checkbox(
                            label=t("detailed_label"), value=False,
                            elem_id="yt-detailed",
                        )
                    with gr.Column(elem_classes=["action-btn"]):
                        yt_btn = gr.Button(t("analyze_btn"), variant="primary")
                with gr.Column(scale=2):
                    yt_gallery = gr.Gallery(
                        label=t("gallery_label"),
                        columns=3,
                        height=420,
                        object_fit="cover",
                        elem_id="yt-gallery",
                    )
            with gr.Row():
                yt_results = gr.Markdown(
                    value=f"*{t('results_placeholder')}*",
                )
            with gr.Accordion("JSON", open=False):
                yt_json = gr.Code(language="json", value="{}", lines=12)

        # ── Panel 2: Image Upload ──
        with gr.Column(visible=False) as panel_image:
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label=t("upload_label"), type="numpy",
                        elem_id="img-input",
                    )
                    with gr.Row():
                        img_disease = gr.Checkbox(
                            label=t("disease_label"), value=False,
                            elem_id="img-disease",
                        )
                        img_detailed = gr.Checkbox(
                            label=t("detailed_label"), value=False,
                            elem_id="img-detailed",
                        )
                    with gr.Column(elem_classes=["action-btn"]):
                        img_btn = gr.Button(t("detect_btn"), variant="primary")
                with gr.Column(scale=1):
                    img_output = gr.Image(
                        label=t("result_image_label"), type="numpy",
                        elem_id="img-output",
                    )
            with gr.Row():
                img_results = gr.Markdown(
                    value=f"*{t('upload_placeholder')}*",
                )
            with gr.Accordion("JSON", open=False):
                img_json = gr.Code(language="json", value="{}", lines=12)

        # ── Panel 3: About ──
        with gr.Column(visible=False) as panel_about:
            about_md = gr.Markdown(
                build_about_md("en"), elem_id="about-section",
            )

        footer_md = gr.Markdown(t("footer"), elem_id="footer-text")

        # ── Tab switching events ──
        # JS: nudge slider value to force Gradio to re-render track fill
        SLIDER_FIX_JS = """
        () => {
            setTimeout(() => {
                document.querySelectorAll('input[type=range]').forEach(el => {
                    const orig = el.value;
                    el.value = Number(orig) + 1;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    requestAnimationFrame(() => {
                        el.value = orig;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                    });
                });
            }, 80);
        }
        """
        tab_outputs = [
            panel_video, panel_image, panel_about,
            tab_btn_video, tab_btn_image, tab_btn_about,
        ]
        tab_btn_video.click(select_tab_0, outputs=tab_outputs, js=SLIDER_FIX_JS)
        tab_btn_image.click(select_tab_1, outputs=tab_outputs)
        tab_btn_about.click(select_tab_2, outputs=tab_outputs)

        # ── Detection events ──
        yt_btn.click(
            handle_youtube,
            [yt_url, yt_frames, yt_disease, yt_detailed, lang],
            [yt_gallery, yt_results, yt_json],
        )
        img_btn.click(
            handle_image_upload,
            [img_input, img_disease, img_detailed, lang],
            [img_output, img_results, img_json],
        )

        # ── Language switch — updates every visible element ──
        lang.change(
            fn=switch_language,
            inputs=[lang],
            outputs=[
                title_md, desc_md,
                yt_url, yt_frames, yt_disease, yt_detailed, yt_btn, yt_gallery, yt_results,
                img_input, img_disease, img_detailed, img_btn, img_output, img_results,
                about_md, footer_md,
                tab_btn_video, tab_btn_image, tab_btn_about,
            ],
        )

        # ── On-load JS: fix label for-attribute mismatches ──
        demo.load(fn=None, js="""
        () => {
            setTimeout(() => {
                document.querySelectorAll('.gradio-container label[for]').forEach(lbl => {
                    const forId = lbl.getAttribute('for');
                    if (!document.getElementById(forId)) {
                        const container = lbl.closest('.block, .form, [class*=wrap]');
                        if (container) {
                            const input = container.querySelector(
                                'input, textarea, select, [role=listbox]'
                            );
                            if (input) {
                                if (!input.id) input.id = 'fix-' + Math.random().toString(36).slice(2, 8);
                                lbl.setAttribute('for', input.id);
                            }
                        }
                    }
                });
                document.querySelectorAll(
                    'input:not([id]):not([name]), textarea:not([id]):not([name]), select:not([id]):not([name])'
                ).forEach(el => {
                    el.id = 'field-' + Math.random().toString(36).slice(2, 8);
                });
            }, 500);
        }
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Strawberry VLA Demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("  Strawberry VLA — Demo Interface")
    print("=" * 50)

    if args.preload:
        print("\nLoading YOLO11...")
        get_yolo()
        print("  Ready")
        print("\nLoading Qwen 2.5 VL...")
        get_qwen()
        print("  Ready")

    demo = build_demo()
    favicon = Path(__file__).parent / "favicon.png"
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        favicon_path=str(favicon) if favicon.exists() else None,
    )


if __name__ == "__main__":
    main()