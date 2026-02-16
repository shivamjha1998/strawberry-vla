"""
strawberry_detector.py — Strawberry Detection Pipeline (Task 1.3)

Uses Qwen 2.5 VL as the primary detector + analyzer:
  - Bounding box detection (Qwen VL native grounding)
  - Ripeness assessment via RGB analysis
  - Disease detection (powdery mildew, anthracnose)

YOLO is included as an optional secondary detector for when
a strawberry-specific fine-tuned model becomes available.

Usage:
    # Process all frames in a directory
    python strawberry_detector.py --frames frames/VIDEO_NAME/

    # Process a single image
    python strawberry_detector.py --image path/to/frame.jpg

    # Process all videos' frames
    python strawberry_detector.py --frames-root frames/

    # With visualization output
    python strawberry_detector.py --frames frames/VIDEO_NAME/ --visualize

    # Skip YOLO (Qwen VL only — faster, less memory)
    python strawberry_detector.py --frames frames/VIDEO_NAME/ --no-yolo
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np


# ─── Prompts ─────────────────────────────────────────────────────

DETECTION_PROMPT = """Detect all strawberries in this image. For each strawberry, output the bounding box coordinates in absolute pixel coordinates and a description.

Output ONLY a JSON array in this exact format, with no other text:
[
  {
    "bbox_2d": [x1, y1, x2, y2],
    "label": "strawberry",
    "ripeness": "green|white|turning|ripe|overripe",
    "color_description": "brief color description",
    "size": "small|medium|large",
    "confidence": "high|medium|low"
  }
]

If no strawberries are visible, output: []

Important:
- Coordinates must be in absolute pixel values (not normalized)
- x1,y1 is top-left corner, x2,y2 is bottom-right corner
- Include ALL visible strawberries, even partially occluded ones
- Be accurate with ripeness: green=unripe, white=early, turning=partially red, ripe=fully red, overripe=dark/mushy"""


DISEASE_PROMPT = """Analyze this image of strawberries for signs of disease.

Look specifically for:
1. Powdery mildew: white/gray powdery coating on leaves or fruit
2. Anthracnose: dark, sunken lesions on fruit; may have pink/orange spore masses
3. Botrytis (gray mold): fuzzy gray mold on fruit
4. Leaf spot: dark spots on leaves

Output ONLY a JSON object in this exact format:
{
  "diseases_detected": [
    {
      "disease": "disease name",
      "severity": "none|mild|moderate|severe",
      "location": "description of where on the plant",
      "confidence": "high|medium|low"
    }
  ],
  "overall_plant_health": "healthy|mild_issues|moderate_issues|severe_issues",
  "notes": "brief observation"
}

If the plants look healthy, output diseases_detected as an empty array."""


RIPENESS_DETAIL_PROMPT = """Analyze the ripeness of the strawberry(ies) in this cropped image region.

Provide a detailed RGB-based ripeness assessment. Output ONLY JSON:
{
  "dominant_color_rgb": [R, G, B],
  "ripeness_stage": "green|white|turning|nearly_ripe|ripe|overripe",
  "ripeness_percentage": 0-100,
  "color_uniformity": "uniform|partially_uniform|uneven",
  "harvest_ready": true/false,
  "notes": "brief observation about color distribution"
}"""


# ─── Model Loaders ───────────────────────────────────────────────

class QwenVLDetector:
    """Qwen 2.5 VL based strawberry detector and analyzer."""

    def __init__(self, model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.config = None

    def load(self):
        """Load model into memory."""
        if self.model is not None:
            return

        print(f"  Loading Qwen 2.5 VL ({self.model_path})...")
        load_start = time.time()

        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self.model, self.processor = load(self.model_path)
        self.config = load_config(self.model_path)
        self._generate = generate
        self._apply_chat_template = apply_chat_template

        print(f"  ✓ Model loaded in {time.time() - load_start:.1f}s")

    def unload(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            print("  ✓ Qwen VL unloaded")

    def _run_inference(self, image_path: str, prompt: str, max_tokens: int = 1000) -> str:
        """Run single inference and return raw text output."""
        self.load()

        formatted_prompt = self._apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )

        output = self._generate(
            self.model, self.processor, formatted_prompt, [image_path],
            max_tokens=max_tokens, verbose=False
        )

        # mlx-vlm 0.3.x returns a GenerationResult object instead of str
        if not isinstance(output, str):
            output = output.text if hasattr(output, 'text') else str(output)

        return output

    def detect_strawberries(self, image_path: str) -> list:
        """
        Detect strawberries in an image using Qwen VL grounding.
        Returns list of detection dicts with bbox and attributes.
        """
        raw_output = self._run_inference(image_path, DETECTION_PROMPT)
        detections = self._parse_json_response(raw_output, expected_type=list, default=[])

        # Get image dimensions for validation
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            detections = self._validate_bboxes(detections, w, h)

        return detections

    def assess_disease(self, image_path: str) -> dict:
        """Analyze image for strawberry diseases."""
        raw_output = self._run_inference(image_path, DISEASE_PROMPT)
        result = self._parse_json_response(raw_output, expected_type=dict, default={
            "diseases_detected": [],
            "overall_plant_health": "unknown",
            "notes": "Could not parse response"
        })
        return result

    def assess_ripeness_detail(self, image_path: str) -> dict:
        """Detailed ripeness assessment for a cropped strawberry region."""
        raw_output = self._run_inference(image_path, RIPENESS_DETAIL_PROMPT, max_tokens=500)
        result = self._parse_json_response(raw_output, expected_type=dict, default={
            "ripeness_stage": "unknown",
            "ripeness_percentage": -1,
            "notes": "Could not parse response"
        })
        return result

    @staticmethod
    def _parse_json_response(text: str, expected_type=list, default=None):
        """Extract JSON from model output, handling common formatting issues."""
        if default is None:
            default = [] if expected_type == list else {}

        # Try direct parse first
        text = text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text (model sometimes adds explanation)
        # Look for [...] or {...}
        if expected_type == list:
            match = re.search(r'\[.*\]', text, re.DOTALL)
        else:
            match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, expected_type):
                    return result
            except json.JSONDecodeError:
                pass

        # Try to fix common issues (trailing commas, etc.)
        cleaned = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            result = json.loads(cleaned)
            if isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass

        print(f"    ⚠ Could not parse JSON from response: {text[:200]}...")
        return default

    @staticmethod
    def _validate_bboxes(detections: list, img_width: int, img_height: int) -> list:
        """Validate and clamp bounding boxes to image dimensions."""
        valid = []
        for det in detections:
            if "bbox_2d" not in det:
                continue

            bbox = det["bbox_2d"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except (ValueError, TypeError):
                continue

            # Clamp to image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            # Ensure valid box (min area)
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            det["bbox_2d"] = [int(x1), int(y1), int(x2), int(y2)]
            valid.append(det)

        return valid


class YOLODetector:
    """YOLO-based strawberry detector (requires fine-tuned model)."""

    def __init__(self, model_path="yolov8n.pt"):
        self.model_path = model_path
        self.model = None

    def load(self):
        if self.model is not None:
            return
        print(f"  Loading YOLO ({self.model_path})...")
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        print(f"  ✓ YOLO loaded")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            print("  ✓ YOLO unloaded")

    def detect(self, image_path: str, conf_threshold: float = 0.3) -> list:
        """
        Run YOLO detection.
        Note: Base COCO model won't detect 'strawberry'.
        Returns detections in same format as QwenVL for compatibility.
        """
        self.load()
        results = self.model(image_path, verbose=False, conf=conf_threshold)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls)]
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()

                detections.append({
                    "bbox_2d": [int(v) for v in xyxy],
                    "label": cls_name,
                    "confidence_score": round(conf, 3),
                    "source": "yolo"
                })

        return detections


# ─── RGB Analysis ────────────────────────────────────────────────

def analyze_rgb_ripeness(image: np.ndarray, bbox: list) -> dict:
    """
    Analyze ripeness based on RGB values within a bounding box.
    Pure computer-vision approach (no model needed).
    """
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return {"error": "empty crop"}

    # Convert to RGB (OpenCV loads as BGR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Mean RGB
    mean_r, mean_g, mean_b = rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()

    # Red ratio: how much red vs green (key ripeness indicator)
    red_ratio = mean_r / (mean_g + 1e-6)

    # HSV analysis for better color segmentation
    mean_h, mean_s, mean_v = hsv[:, :, 0].mean(), hsv[:, :, 1].mean(), hsv[:, :, 2].mean()

    # Red pixel percentage (hue 0-10 or 170-180 in OpenCV HSV)
    red_mask_low = (hsv[:, :, 0] < 10) & (hsv[:, :, 1] > 50)
    red_mask_high = (hsv[:, :, 0] > 170) & (hsv[:, :, 1] > 50)
    red_pixel_pct = ((red_mask_low | red_mask_high).sum() / crop[:, :, 0].size) * 100

    # Green pixel percentage (hue 35-85)
    green_mask = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 30)
    green_pixel_pct = (green_mask.sum() / crop[:, :, 0].size) * 100

    # White pixel percentage (low saturation, high value)
    white_mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 180)
    white_pixel_pct = (white_mask.sum() / crop[:, :, 0].size) * 100

    # Determine ripeness stage
    if red_pixel_pct > 60:
        if mean_v < 80:
            stage = "overripe"
            pct = 100
        else:
            stage = "ripe"
            pct = 85 + min(15, red_pixel_pct - 60) * 0.25
    elif red_pixel_pct > 30:
        stage = "turning"
        pct = 40 + (red_pixel_pct - 30) * 1.5
    elif white_pixel_pct > 30:
        stage = "white"
        pct = 20 + white_pixel_pct * 0.4
    elif green_pixel_pct > 30:
        stage = "green"
        pct = max(0, 20 - green_pixel_pct * 0.3)
    else:
        stage = "turning"
        pct = 50

    return {
        "mean_rgb": [int(mean_r), int(mean_g), int(mean_b)],
        "mean_hsv": [int(mean_h), int(mean_s), int(mean_v)],
        "red_pixel_pct": round(red_pixel_pct, 1),
        "green_pixel_pct": round(green_pixel_pct, 1),
        "white_pixel_pct": round(white_pixel_pct, 1),
        "red_green_ratio": round(red_ratio, 2),
        "ripeness_stage": stage,
        "ripeness_percentage": round(min(100, max(0, pct)), 1),
        "harvest_ready": stage in ("ripe",)
    }


# ─── Visualization ───────────────────────────────────────────────

# Color scheme for ripeness stages
RIPENESS_COLORS = {
    "green": (0, 180, 0),       # Green
    "white": (200, 200, 200),   # Light gray
    "turning": (0, 165, 255),   # Orange
    "nearly_ripe": (0, 100, 255),  # Dark orange
    "ripe": (0, 0, 255),        # Red
    "overripe": (0, 0, 139),    # Dark red
    "unknown": (128, 128, 128)  # Gray
}


def visualize_detections(image_path: str, detections: list, output_path: str,
                         rgb_analyses: list = None):
    """Draw bounding boxes and labels on an image."""
    img = cv2.imread(image_path)
    if img is None:
        return

    for i, det in enumerate(detections):
        bbox = det.get("bbox_2d", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        ripeness = det.get("ripeness", "unknown")
        color = RIPENESS_COLORS.get(ripeness, (128, 128, 128))
        # Convert RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

        # Label
        label_parts = [f"#{i+1}"]
        if ripeness != "unknown":
            label_parts.append(ripeness)

        # Add RGB ripeness if available
        if rgb_analyses and i < len(rgb_analyses):
            rgb_data = rgb_analyses[i]
            if "ripeness_percentage" in rgb_data:
                label_parts.append(f"{rgb_data['ripeness_percentage']:.0f}%")

        label = " | ".join(label_parts)

        # Label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color_bgr, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Summary text at top
    summary = f"Detected: {len(detections)} strawberries"
    cv2.putText(img, summary, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ─── Pipeline ────────────────────────────────────────────────────

def process_frame(
    image_path: str,
    qwen_detector: QwenVLDetector,
    yolo_detector: Optional[YOLODetector] = None,
    do_disease: bool = False,
    do_rgb: bool = True
) -> dict:
    """
    Full detection pipeline for a single frame.

    Returns structured result dict.
    """
    result = {
        "image_path": image_path,
        "timestamp": Path(image_path).stem,  # e.g., frame_00042_t42.00s
        "detections": [],
        "rgb_analyses": [],
        "disease_assessment": None,
        "yolo_detections": [],
        "processing_time": {}
    }

    # --- Qwen VL Detection ---
    t0 = time.time()
    detections = qwen_detector.detect_strawberries(image_path)
    result["detections"] = detections
    result["processing_time"]["qwen_detection"] = round(time.time() - t0, 2)
    print(f"    Qwen VL: {len(detections)} strawberries ({result['processing_time']['qwen_detection']}s)")

    # --- RGB Analysis ---
    if do_rgb and detections:
        t0 = time.time()
        img = cv2.imread(image_path)
        if img is not None:
            for det in detections:
                bbox = det.get("bbox_2d", [])
                if len(bbox) == 4:
                    rgb_result = analyze_rgb_ripeness(img, bbox)
                    result["rgb_analyses"].append(rgb_result)
                else:
                    result["rgb_analyses"].append({"error": "invalid bbox"})
        result["processing_time"]["rgb_analysis"] = round(time.time() - t0, 4)

    # --- Disease Assessment ---
    if do_disease:
        t0 = time.time()
        disease_result = qwen_detector.assess_disease(image_path)
        result["disease_assessment"] = disease_result
        result["processing_time"]["disease"] = round(time.time() - t0, 2)

    # --- YOLO Detection (optional, for comparison) ---
    if yolo_detector is not None:
        t0 = time.time()
        yolo_dets = yolo_detector.detect(image_path)
        result["yolo_detections"] = yolo_dets
        result["processing_time"]["yolo"] = round(time.time() - t0, 3)

    return result


def process_frame_directory(
    frame_dir: str,
    output_dir: str,
    qwen_detector: QwenVLDetector,
    yolo_detector: Optional[YOLODetector] = None,
    visualize: bool = True,
    do_disease: bool = False,
    max_frames: int = 0,
    frame_step: int = 1
) -> dict:
    """Process all frames in a directory."""
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("frame_*.jpg"))

    if not frames:
        print(f"  No frames found in {frame_dir}")
        return {"frame_count": 0, "results": []}

    # Apply frame step (skip frames for faster processing)
    if frame_step > 1:
        frames = frames[::frame_step]
    if max_frames > 0:
        frames = frames[:max_frames]

    print(f"\n  Processing {len(frames)} frames from {frame_dir.name}...")

    # Create output directory
    video_output_dir = Path(output_dir) / frame_dir.name
    os.makedirs(video_output_dir, exist_ok=True)
    if visualize:
        os.makedirs(video_output_dir / "annotated", exist_ok=True)

    results = []
    total_strawberries = 0

    for i, frame_path in enumerate(frames):
        print(f"\n  [{i+1}/{len(frames)}] {frame_path.name}")

        # Process frame
        result = process_frame(
            str(frame_path), qwen_detector, yolo_detector,
            do_disease=do_disease
        )
        results.append(result)
        total_strawberries += len(result["detections"])

        # Visualize
        if visualize and result["detections"]:
            viz_path = str(video_output_dir / "annotated" / f"det_{frame_path.name}")
            visualize_detections(
                str(frame_path), result["detections"], viz_path,
                result.get("rgb_analyses")
            )

    # Save results
    summary = {
        "source_dir": str(frame_dir),
        "frames_processed": len(frames),
        "total_strawberries_detected": total_strawberries,
        "avg_strawberries_per_frame": round(total_strawberries / max(1, len(frames)), 2),
        "processed_at": datetime.now().isoformat(),
        "results": results
    }

    results_path = video_output_dir / "detection_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  ✓ Results saved: {results_path}")
    print(f"  ✓ Total: {total_strawberries} strawberries in {len(frames)} frames")

    return summary


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Strawberry VLA — Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input
    parser.add_argument("--image", type=str, help="Single image to process")
    parser.add_argument("--frames", type=str, help="Directory of frames to process")
    parser.add_argument("--frames-root", type=str, help="Root directory containing multiple frame dirs")

    # Processing options
    parser.add_argument("--no-yolo", action="store_true", help="Skip YOLO detection")
    parser.add_argument("--disease", action="store_true", help="Enable disease detection")
    parser.add_argument("--visualize", action="store_true", help="Save annotated images")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process per video (0=all)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Process every Nth frame (default: 1=all)")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory (default: outputs)")

    # Model
    parser.add_argument("--model", type=str,
                        default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
                        help="Qwen VL model path")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="YOLO model path")

    args = parser.parse_args()

    if not args.image and not args.frames and not args.frames_root:
        parser.print_help()
        print("\nError: Provide --image, --frames, or --frames-root")
        sys.exit(1)

    # Initialize detectors
    qwen = QwenVLDetector(args.model)
    yolo = None if args.no_yolo else YOLODetector(args.yolo_model)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        # --- Single image mode ---
        print(f"\n{'='*60}")
        print(f"Processing single image: {args.image}")
        print(f"{'='*60}")

        qwen.load()
        result = process_frame(
            args.image, qwen, yolo, do_disease=args.disease
        )

        # Print results
        print(f"\n{'─'*60}")
        print(f"Detections: {len(result['detections'])}")
        for i, det in enumerate(result["detections"]):
            print(f"  #{i+1}: {det.get('ripeness', '?')} @ {det.get('bbox_2d', '?')}")
            if i < len(result.get("rgb_analyses", [])):
                rgb = result["rgb_analyses"][i]
                print(f"       RGB: {rgb.get('mean_rgb', '?')}, "
                      f"Ripeness: {rgb.get('ripeness_percentage', '?')}%, "
                      f"Harvest: {rgb.get('harvest_ready', '?')}")

        if result.get("disease_assessment"):
            print(f"\nDisease: {json.dumps(result['disease_assessment'], indent=2)}")

        # Visualize
        if args.visualize:
            viz_path = os.path.join(args.output_dir, f"detected_{Path(args.image).name}")
            visualize_detections(args.image, result["detections"], viz_path,
                                result.get("rgb_analyses"))
            print(f"\n✓ Annotated image: {viz_path}")

        # Save JSON
        json_path = os.path.join(args.output_dir, "single_detection.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"✓ Results: {json_path}")

    elif args.frames:
        # --- Single video frames ---
        qwen.load()
        summary = process_frame_directory(
            args.frames, args.output_dir, qwen, yolo,
            visualize=args.visualize, do_disease=args.disease,
            max_frames=args.max_frames, frame_step=args.frame_step
        )

    elif args.frames_root:
        # --- All videos ---
        root = Path(args.frames_root)
        subdirs = [d for d in sorted(root.iterdir())
                    if d.is_dir() and not d.name.startswith(".")]

        print(f"\n{'='*60}")
        print(f"Processing {len(subdirs)} video frame sets")
        print(f"{'='*60}")

        qwen.load()

        all_summaries = []
        for subdir in subdirs:
            summary = process_frame_directory(
                str(subdir), args.output_dir, qwen, yolo,
                visualize=args.visualize, do_disease=args.disease,
                max_frames=args.max_frames, frame_step=args.frame_step
            )
            all_summaries.append(summary)

        # Overall summary
        total_frames = sum(s.get("frames_processed", 0) for s in all_summaries)
        total_strawberries = sum(s.get("total_strawberries_detected", 0) for s in all_summaries)

        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"  Videos processed: {len(all_summaries)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Total strawberries: {total_strawberries}")
        print(f"  Output: {args.output_dir}/")

    # Cleanup
    qwen.unload()
    if yolo:
        yolo.unload()


if __name__ == "__main__":
    main()
