"""
preview_frames.py — Quick preview utility for extracted frames

Shows frame count, resolution, sample images, and RGB statistics
for quality-checking the ingestion pipeline output.

Usage:
    python preview_frames.py                    # Preview all frame directories
    python preview_frames.py --dir frames/VIDEO_NAME  # Preview specific video's frames
    python preview_frames.py --sample 5         # Show 5 sample frames per video
    python preview_frames.py --create-montage   # Create a montage image of sample frames
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def analyze_frame(frame_path: str) -> dict:
    """Get basic statistics for a single frame."""
    img = cv2.imread(frame_path)
    if img is None:
        return {"error": "could not read"}

    h, w, c = img.shape
    # Convert to RGB for analysis
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return {
        "resolution": f"{w}x{h}",
        "mean_rgb": [int(rgb[:, :, i].mean()) for i in range(3)],
        "std_rgb": [int(rgb[:, :, i].std()) for i in range(3)],
        "brightness": int(rgb.mean()),
        "filesize_kb": os.path.getsize(frame_path) // 1024
    }


def preview_directory(frame_dir: str, sample_count: int = 3, create_montage: bool = False):
    """Preview frames in a directory."""
    frame_dir = Path(frame_dir)

    if not frame_dir.exists():
        print(f"  ✗ Directory not found: {frame_dir}")
        return

    # Load metadata if available
    metadata_path = frame_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        print(f"  Video: {meta.get('video_name', 'unknown')}")
        print(f"  Original: {meta.get('original_resolution', '?')}")
        print(f"  Extracted: {meta.get('extracted_resolution', '?')}")
        print(f"  Duration: {meta.get('duration_seconds', '?')}s")
        print(f"  FPS: {meta.get('extraction_fps', '?')}")
    else:
        print(f"  (No metadata.json found)")

    # Count frames
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    print(f"  Total frames: {len(frames)}")

    if not frames:
        return

    # Sample frames evenly
    if len(frames) <= sample_count:
        samples = frames
    else:
        indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        samples = [frames[i] for i in indices]

    # Analyze samples
    print(f"\n  Sample frames ({len(samples)}):")
    for frame_path in samples:
        stats = analyze_frame(str(frame_path))
        if "error" in stats:
            print(f"    {frame_path.name}: ERROR")
            continue
        print(f"    {frame_path.name}: {stats['resolution']}, "
              f"brightness={stats['brightness']}, "
              f"RGB=({stats['mean_rgb'][0]},{stats['mean_rgb'][1]},{stats['mean_rgb'][2]}), "
              f"{stats['filesize_kb']}KB")

    # Create montage
    if create_montage and len(samples) > 0:
        montage_path = frame_dir / "montage_preview.jpg"
        imgs = []
        target_h = 300

        for sp in samples:
            img = cv2.imread(str(sp))
            if img is not None:
                scale = target_h / img.shape[0]
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, target_h))
                # Add timestamp label
                name = sp.stem
                cv2.putText(img, name, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                imgs.append(img)

        if imgs:
            # Make all images same height, concatenate horizontally
            montage = np.hstack(imgs)
            cv2.imwrite(str(montage_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"\n  ✓ Montage saved: {montage_path}")


def main():
    parser = argparse.ArgumentParser(description="Preview extracted frames")
    parser.add_argument("--dir", type=str, default=None,
                        help="Specific frame directory to preview")
    parser.add_argument("--frame-root", type=str, default="frames",
                        help="Root frames directory (default: frames)")
    parser.add_argument("--sample", type=int, default=3,
                        help="Number of sample frames per video (default: 3)")
    parser.add_argument("--create-montage", action="store_true",
                        help="Create a montage image of sample frames")

    args = parser.parse_args()

    if args.dir:
        # Preview specific directory
        print(f"\n{'='*60}")
        print(f"Previewing: {args.dir}")
        print(f"{'='*60}")
        preview_directory(args.dir, args.sample, args.create_montage)

    else:
        # Preview all directories under frame root
        root = Path(args.frame_root)
        if not root.exists():
            print(f"Frame root not found: {root}")
            print("Run video_ingest.py first to extract frames.")
            return

        subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]

        if not subdirs:
            print(f"No frame directories found in {root}")
            return

        print(f"\n{'='*60}")
        print(f"Frame Preview — {len(subdirs)} videos")
        print(f"{'='*60}")

        total_frames = 0
        for subdir in subdirs:
            print(f"\n{'─'*60}")
            print(f"📁 {subdir.name}")
            print(f"{'─'*60}")
            preview_directory(str(subdir), args.sample, args.create_montage)
            frame_count = len(list(subdir.glob("frame_*.jpg")))
            total_frames += frame_count

        print(f"\n{'='*60}")
        print(f"Total: {len(subdirs)} videos, {total_frames} frames")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
