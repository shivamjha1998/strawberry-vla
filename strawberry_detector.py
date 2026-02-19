"""
strawberry_detector.py — Strawberry Detection Pipeline (Updated)

NEW Architecture (YOLO11 trained + Qwen VL analysis):
  Frame → YOLO11 (fast, <10ms) → bounding boxes
       → Each crop → RGB analysis (instant)
       → Each crop → Qwen VL (detailed, optional) → ripeness/disease

Usage:
    # Fast mode (YOLO + RGB only)
    python strawberry_detector.py --image frame.jpg --visualize

    # Detailed mode (YOLO + RGB + Qwen VL per crop)
    python strawberry_detector.py --image frame.jpg --detailed --visualize

    # Process frame directory
    python strawberry_detector.py --frames frames/VIDEO/ --visualize

    # With disease detection
    python strawberry_detector.py --frames frames/VIDEO/ --disease --visualize

    # Legacy: Qwen VL as primary detector (slow, like before training)
    python strawberry_detector.py --image frame.jpg --legacy
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

import cv2
import numpy as np


# ─── Prompts ─────────────────────────────────────────────────────────────────

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

RIPENESS_DETAIL_PROMPT = """Analyze the ripeness of the strawberry in this cropped image.

Provide a detailed ripeness assessment. Output ONLY JSON:
{
  "ripeness_stage": "green|white|turning|nearly_ripe|ripe|overripe",
  "ripeness_percentage": 0-100,
  "color_uniformity": "uniform|partially_uniform|uneven",
  "surface_quality": "smooth|slightly_rough|damaged",
  "harvest_ready": true/false,
  "notes": "brief observation about color, texture, and condition"
}"""


# ─── YOLO11 Detector (PRIMARY) ───────────────────────────────────────────────

def _get_device():
    """Return the best available accelerator: mps > cuda > cpu."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class YOLODetector:
    """YOLO11 strawberry detector — trained on harvesting robot dataset."""

    def __init__(self, model_path="strawberry_yolo_best.pt"):
        self.model_path = model_path
        self.model = None

    def load(self):
        if self.model is not None:
            return
        print(f"  Loading YOLO11 ({self.model_path})...")
        t0 = time.time()
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        print(f"  ✓ YOLO11 loaded in {time.time() - t0:.1f}s — classes: {self.model.names}")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            print("  ✓ YOLO11 unloaded")

    def detect(self, image_path: str, conf_threshold: float = 0.3) -> list:
        self.load()
        results = self.model(image_path, verbose=False, conf=conf_threshold, device=_get_device())

        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                ripeness = self._map_class_to_ripeness(cls_name)

                detections.append({
                    "bbox_2d": [int(v) for v in xyxy],
                    "label": "strawberry",
                    "yolo_class": cls_name,
                    "ripeness": ripeness,
                    "confidence_score": round(conf, 3),
                    "confidence": "high" if conf > 0.7 else "medium" if conf > 0.4 else "low",
                    "source": "yolo11"
                })

        return detections

    @staticmethod
    def _map_class_to_ripeness(cls_name: str) -> str:
        name = cls_name.lower().replace("-", "_").replace(" ", "_")
        if "ripe" in name and "unripe" not in name:
            return "ripe"
        elif "unripe" in name or "green" in name:
            return "green"
        elif "turning" in name or "partial" in name:
            return "turning"
        elif "flower" in name:
            return "flower"
        elif "overripe" in name:
            return "overripe"
        return "unknown"


# ─── Qwen VL Detector (SECONDARY — analysis) ─────────────────────────────────

class QwenVLDetector:
    """Qwen 2.5 VL — now used for detailed crop analysis & disease detection."""

    def __init__(self, model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.config = None

    def load(self):
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
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            print("  ✓ Qwen VL unloaded")

    def _run_inference(self, image_path: str, prompt: str, max_tokens: int = 1000) -> str:
        self.load()
        formatted_prompt = self._apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )
        result = self._generate(
            self.model, self.processor, formatted_prompt, [image_path],
            max_tokens=max_tokens, verbose=False
        )
        # mlx_vlm >= 0.1.x returns a GenerationResult object, not a string
        if isinstance(result, str):
            return result
        return getattr(result, "text", str(result))

    def detect_strawberries(self, image_path: str) -> list:
        """Legacy: full-frame detection via Qwen VL (slow)."""
        raw_output = self._run_inference(image_path, DETECTION_PROMPT)
        detections = self._parse_json_response(raw_output, expected_type=list, default=[])
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            detections = self._validate_bboxes(detections, w, h)
        for det in detections:
            det["source"] = "qwen_vl"
        return detections

    def assess_disease(self, image_path: str) -> dict:
        raw_output = self._run_inference(image_path, DISEASE_PROMPT)
        return self._parse_json_response(raw_output, expected_type=dict, default={
            "diseases_detected": [], "overall_plant_health": "unknown",
            "notes": "Could not parse response"
        })

    def assess_ripeness_detail(self, image_path: str) -> dict:
        raw_output = self._run_inference(image_path, RIPENESS_DETAIL_PROMPT, max_tokens=500)
        return self._parse_json_response(raw_output, expected_type=dict, default={
            "ripeness_stage": "unknown", "ripeness_percentage": -1,
            "notes": "Could not parse response"
        })

    @staticmethod
    def _parse_json_response(text, expected_type=list, default=None):
        if default is None:
            default = [] if expected_type == list else {}
        text = text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass
        pattern = r'\[.*\]' if expected_type == list else r'\{.*\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, expected_type):
                    return result
            except json.JSONDecodeError:
                pass
        cleaned = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            result = json.loads(cleaned)
            if isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass
        print(f"    ⚠ Could not parse JSON: {text[:200]}...")
        return default

    @staticmethod
    def _validate_bboxes(detections, img_width, img_height):
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
            x1, y1 = max(0, min(x1, img_width)), max(0, min(y1, img_height))
            x2, y2 = max(0, min(x2, img_width)), max(0, min(y2, img_height))
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue
            det["bbox_2d"] = [int(x1), int(y1), int(x2), int(y2)]
            valid.append(det)
        return valid


# ─── RGB Analysis ─────────────────────────────────────────────────────────────

def analyze_rgb_ripeness(image: np.ndarray, bbox: list) -> dict:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return {"error": "empty crop"}

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mean_r, mean_g, mean_b = rgb[:,:,0].mean(), rgb[:,:,1].mean(), rgb[:,:,2].mean()
    red_ratio = mean_r / (mean_g + 1e-6)
    mean_h, mean_s, mean_v = hsv[:,:,0].mean(), hsv[:,:,1].mean(), hsv[:,:,2].mean()

    red_mask = ((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)) & (hsv[:,:,1] > 50)
    red_pixel_pct = (red_mask.sum() / crop[:,:,0].size) * 100
    green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 30)
    green_pixel_pct = (green_mask.sum() / crop[:,:,0].size) * 100
    white_mask = (hsv[:,:,1] < 40) & (hsv[:,:,2] > 180)
    white_pixel_pct = (white_mask.sum() / crop[:,:,0].size) * 100

    if red_pixel_pct > 60:
        stage, pct = ("overripe", 100) if mean_v < 80 else ("ripe", 85 + min(15, red_pixel_pct - 60) * 0.25)
    elif red_pixel_pct > 30:
        stage, pct = "turning", 40 + (red_pixel_pct - 30) * 1.5
    elif white_pixel_pct > 30:
        stage, pct = "white", 20 + white_pixel_pct * 0.4
    elif green_pixel_pct > 30:
        stage, pct = "green", max(0, 20 - green_pixel_pct * 0.3)
    else:
        stage, pct = "turning", 50

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


# ─── Visualization ────────────────────────────────────────────────────────────

RIPENESS_COLORS = {
    "green": (0, 180, 0), "white": (200, 200, 200), "turning": (0, 165, 255),
    "nearly_ripe": (0, 100, 255), "ripe": (0, 0, 255), "overripe": (0, 0, 139),
    "flower": (255, 200, 0), "unknown": (128, 128, 128)
}

def visualize_detections(image_path, detections, output_path, rgb_analyses=None):
    img = cv2.imread(image_path)
    if img is None:
        return

    for i, det in enumerate(detections):
        bbox = det.get("bbox_2d", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        ripeness = det.get("ripeness", "unknown")
        color_bgr = tuple(reversed(RIPENESS_COLORS.get(ripeness, (128, 128, 128))))

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

        parts = [f"#{i+1}"]
        yolo_cls = det.get("yolo_class", "")
        if yolo_cls:
            parts.append(yolo_cls)
        elif ripeness != "unknown":
            parts.append(ripeness)

        conf = det.get("confidence_score")
        if conf is not None:
            parts.append(f"{conf:.0%}")

        if rgb_analyses and i < len(rgb_analyses) and "ripeness_percentage" in rgb_analyses[i]:
            parts.append(f"ripe:{rgb_analyses[i]['ripeness_percentage']:.0f}%")

        label = " | ".join(parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    src = "YOLO11" if any(d.get("source") == "yolo11" for d in detections) else "Qwen VL"
    cv2.putText(img, f"{src}: {len(detections)} strawberries", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def process_frame(image_path, yolo_detector, qwen_detector=None,
                  do_disease=False, do_detailed=False, do_rgb=True,
                  legacy_mode=False):
    result = {
        "image_path": image_path, "timestamp": Path(image_path).stem,
        "detections": [], "rgb_analyses": [], "qwen_analyses": [],
        "disease_assessment": None, "processing_time": {}
    }

    if legacy_mode and qwen_detector:
        t0 = time.time()
        result["detections"] = qwen_detector.detect_strawberries(image_path)
        result["processing_time"]["detection"] = round(time.time() - t0, 2)
        print(f"    Qwen VL: {len(result['detections'])} strawberries ({result['processing_time']['detection']}s)")
    else:
        t0 = time.time()
        result["detections"] = yolo_detector.detect(image_path)
        result["processing_time"]["detection"] = round(time.time() - t0, 4)
        print(f"    YOLO11: {len(result['detections'])} strawberries ({result['processing_time']['detection']*1000:.1f}ms)")

    detections = result["detections"]

    # RGB analysis (instant)
    if do_rgb and detections:
        t0 = time.time()
        img = cv2.imread(image_path)
        if img is not None:
            for det in detections:
                bbox = det.get("bbox_2d", [])
                result["rgb_analyses"].append(
                    analyze_rgb_ripeness(img, bbox) if len(bbox) == 4 else {"error": "invalid bbox"}
                )
        result["processing_time"]["rgb_analysis"] = round(time.time() - t0, 4)

    # Qwen VL detailed crop analysis (optional)
    if do_detailed and qwen_detector and detections:
        t0 = time.time()
        img = cv2.imread(image_path)
        if img is not None:
            import tempfile
            h, w = img.shape[:2]
            for det in detections:
                bbox = det.get("bbox_2d", [])
                if len(bbox) != 4:
                    result["qwen_analyses"].append({"error": "invalid bbox"})
                    continue
                x1, y1, x2, y2 = bbox
                pad_x, pad_y = int((x2-x1)*0.1), int((y2-y1)*0.1)
                crop = img[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
                if crop.size == 0:
                    result["qwen_analyses"].append({"error": "empty crop"})
                    continue
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    cv2.imwrite(f.name, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    crop_path = f.name
                try:
                    result["qwen_analyses"].append(qwen_detector.assess_ripeness_detail(crop_path))
                finally:
                    os.unlink(crop_path)
        result["processing_time"]["qwen_analysis"] = round(time.time() - t0, 2)
        print(f"    Qwen VL analysis: {len(result['qwen_analyses'])} crops ({result['processing_time']['qwen_analysis']}s)")

    # Disease (optional, full frame)
    if do_disease and qwen_detector:
        t0 = time.time()
        result["disease_assessment"] = qwen_detector.assess_disease(image_path)
        result["processing_time"]["disease"] = round(time.time() - t0, 2)

    return result


def process_frame_directory(frame_dir, output_dir, yolo_detector, qwen_detector=None,
                           visualize=True, do_disease=False, do_detailed=False,
                           legacy_mode=False, max_frames=0, frame_step=1):
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"  No frames found in {frame_dir}")
        return {"frame_count": 0, "results": []}

    if frame_step > 1:
        frames = frames[::frame_step]
    if max_frames > 0:
        frames = frames[:max_frames]

    print(f"\n  Processing {len(frames)} frames from {frame_dir.name}...")

    video_output_dir = Path(output_dir) / frame_dir.name
    os.makedirs(video_output_dir, exist_ok=True)
    if visualize:
        os.makedirs(video_output_dir / "annotated", exist_ok=True)

    results, total = [], 0
    for i, fp in enumerate(frames):
        print(f"\n  [{i+1}/{len(frames)}] {fp.name}")
        r = process_frame(str(fp), yolo_detector, qwen_detector,
                          do_disease=do_disease, do_detailed=do_detailed, legacy_mode=legacy_mode)
        results.append(r)
        total += len(r["detections"])
        if visualize and r["detections"]:
            visualize_detections(str(fp), r["detections"],
                                str(video_output_dir / "annotated" / f"det_{fp.name}"),
                                r.get("rgb_analyses"))

    summary = {
        "source_dir": str(frame_dir),
        "detector": "legacy_qwen_vl" if legacy_mode else "yolo11",
        "frames_processed": len(frames),
        "total_strawberries_detected": total,
        "avg_strawberries_per_frame": round(total / max(1, len(frames)), 2),
        "processed_at": datetime.now().isoformat(),
        "results": results
    }
    results_path = video_output_dir / "detection_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  ✓ Results saved: {results_path}")
    print(f"  ✓ Total: {total} strawberries in {len(frames)} frames")
    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Strawberry VLA — Detection Pipeline",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=str, help="Single image to process")
    parser.add_argument("--frames", type=str, help="Directory of frames")
    parser.add_argument("--frames-root", type=str, help="Root dir with multiple frame dirs")
    parser.add_argument("--detailed", action="store_true", help="Qwen VL analysis per crop (slower)")
    parser.add_argument("--disease", action="store_true", help="Disease detection via Qwen VL")
    parser.add_argument("--visualize", action="store_true", help="Save annotated images")
    parser.add_argument("--legacy", action="store_true", help="Qwen VL as primary detector (old slow mode)")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--yolo-model", type=str, default="strawberry_yolo_best.pt")
    parser.add_argument("--qwen-model", type=str, default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    args = parser.parse_args()

    if not args.image and not args.frames and not args.frames_root:
        parser.print_help()
        print("\nError: Provide --image, --frames, or --frames-root")
        sys.exit(1)

    if not args.legacy and not os.path.exists(args.yolo_model):
        print(f"\n⚠ YOLO model not found: {args.yolo_model}")
        print("  Train it: python train_strawberry_yolo.py --download --api-key KEY --train")
        print("  Or use: --legacy (Qwen VL detector, slow)")
        sys.exit(1)

    yolo = YOLODetector(args.yolo_model)
    needs_qwen = args.detailed or args.disease or args.legacy
    qwen = QwenVLDetector(args.qwen_model) if needs_qwen else None
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        print(f"\n{'='*60}\nProcessing: {args.image}\n{'='*60}")
        if not args.legacy: yolo.load()
        if qwen: qwen.load()
        result = process_frame(args.image, yolo, qwen, do_disease=args.disease,
                               do_detailed=args.detailed, legacy_mode=args.legacy)

        print(f"\n{'─'*60}\nDetections: {len(result['detections'])}")
        for i, det in enumerate(result["detections"]):
            cls = det.get("yolo_class", det.get("ripeness", "?"))
            conf = det.get("confidence_score", "?")
            print(f"  #{i+1}: {cls} (conf={conf}) @ {det.get('bbox_2d')}")
            if i < len(result.get("rgb_analyses", [])):
                rgb = result["rgb_analyses"][i]
                if "error" not in rgb:
                    print(f"       RGB: {rgb['mean_rgb']}, ripe={rgb['ripeness_percentage']}%, harvest={rgb['harvest_ready']}")
            if i < len(result.get("qwen_analyses", [])):
                qa = result["qwen_analyses"][i]
                if "error" not in qa:
                    print(f"       Qwen: {qa.get('ripeness_stage')} ({qa.get('ripeness_percentage')}%)")

        t = result["processing_time"]
        det_val = t.get("detection", 0)
        print(f"\n⏱  Detection: {det_val*1000:.1f}ms" if det_val < 1 else f"\n⏱  Detection: {det_val}s")

        if args.visualize:
            viz = os.path.join(args.output_dir, f"detected_{Path(args.image).name}")
            visualize_detections(args.image, result["detections"], viz, result.get("rgb_analyses"))
            print(f"✓ Annotated: {viz}")

        with open(os.path.join(args.output_dir, "single_detection.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

    elif args.frames:
        if not args.legacy: yolo.load()
        if qwen: qwen.load()
        process_frame_directory(args.frames, args.output_dir, yolo, qwen,
                                visualize=args.visualize, do_disease=args.disease,
                                do_detailed=args.detailed, legacy_mode=args.legacy,
                                max_frames=args.max_frames, frame_step=args.frame_step)

    elif args.frames_root:
        root = Path(args.frames_root)
        subdirs = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
        print(f"\n{'='*60}\nProcessing {len(subdirs)} video frame sets\n{'='*60}")
        if not args.legacy: yolo.load()
        if qwen: qwen.load()
        sums = []
        for sd in subdirs:
            sums.append(process_frame_directory(str(sd), args.output_dir, yolo, qwen,
                        visualize=args.visualize, do_disease=args.disease,
                        do_detailed=args.detailed, legacy_mode=args.legacy,
                        max_frames=args.max_frames, frame_step=args.frame_step))
        tf = sum(s.get("frames_processed", 0) for s in sums)
        ts = sum(s.get("total_strawberries_detected", 0) for s in sums)
        print(f"\n{'='*60}\nSUMMARY: {len(sums)} videos, {tf} frames, {ts} strawberries\n{'='*60}")

    yolo.unload()
    if qwen: qwen.unload()


if __name__ == "__main__":
    main()