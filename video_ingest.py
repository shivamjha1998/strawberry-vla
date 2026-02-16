"""
video_ingest.py — Video Ingestion Pipeline for Strawberry VLA

Downloads YouTube videos and extracts frames at configurable FPS.
Frames are organized by video in a structured directory.

Usage:
    # Download and extract from a single URL
    python video_ingest.py --url "https://www.youtube.com/watch?v=XXXX" --fps 1

    # Download and extract from the curated video list
    python video_ingest.py --from-list --fps 1

    # Extract frames from an already-downloaded local video
    python video_ingest.py --local path/to/video.mp4 --fps 2

    # Custom output directory
    python video_ingest.py --url "..." --fps 1 --output-dir ./my_frames
"""

import argparse
import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


# ─── Default Config ──────────────────────────────────────────────
DEFAULT_FPS = 1.0           # Extract 1 frame per second
DEFAULT_VIDEO_DIR = "videos"
DEFAULT_FRAME_DIR = "frames"
DEFAULT_MAX_RESOLUTION = 1280  # Max width/height for extracted frames
DEFAULT_VIDEO_FORMAT = "mp4"
DEFAULT_MAX_DURATION = 600  # Max 10 minutes per video (seconds)


# ─── Curated YouTube Videos ─────────────────────────────────────
# Good strawberry greenhouse videos for testing the VLA pipeline.
# Mix of: close-up fruit, greenhouse walkthrough, harvesting, disease examples
CURATED_VIDEOS = [
    {
        "url": "https://www.youtube.com/watch?v=6rclZrYB_I8",
        "description": "5 Reasons to Grow Strawberries in the Poly-tunnel, NOT Outside.",
        "tags": ["greenhouse", "ripe", "close-up"]
    },
]


# ─── Video Download ──────────────────────────────────────────────

def download_video(url: str, output_dir: str, max_duration: int = DEFAULT_MAX_DURATION) -> dict:
    """
    Download a YouTube video using yt-dlp.

    Returns dict with:
        - filepath: path to downloaded video
        - title: video title
        - duration: video duration in seconds
        - success: bool
    """
    os.makedirs(output_dir, exist_ok=True)

    # First, get video info
    print(f"  Fetching video info...")
    try:
        info_result = subprocess.run(
            [
                "yt-dlp",
                "--print", "%(title)s",
                "--print", "%(duration)s",
                "--print", "%(id)s",
                "--no-download",
                url
            ],
            capture_output=True, text=True, timeout=30
        )

        if info_result.returncode != 0:
            print(f"  ✗ Failed to fetch info: {info_result.stderr.strip()}")
            return {"success": False, "error": info_result.stderr.strip()}

        lines = info_result.stdout.strip().split("\n")
        title = lines[0] if len(lines) > 0 else "unknown"
        duration = int(float(lines[1])) if len(lines) > 1 else 0
        video_id = lines[2] if len(lines) > 2 else "unknown"

        print(f"  Title: {title}")
        print(f"  Duration: {duration}s")

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout fetching video info")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"success": False, "error": str(e)}

    # Sanitize filename
    safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)[:80]
    safe_title = safe_title.strip().replace(" ", "_")
    output_template = os.path.join(output_dir, f"{safe_title}__{video_id}.%(ext)s")

    # Check if already downloaded
    expected_path = os.path.join(output_dir, f"{safe_title}__{video_id}.{DEFAULT_VIDEO_FORMAT}")
    if os.path.exists(expected_path):
        print(f"  ✓ Already downloaded: {expected_path}")
        return {
            "success": True,
            "filepath": expected_path,
            "title": title,
            "duration": duration,
            "video_id": video_id,
            "skipped": True
        }

    # Download
    print(f"  Downloading (max {max_duration}s)...")
    download_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--merge-output-format", DEFAULT_VIDEO_FORMAT,
        "--download-sections", f"*0-{max_duration}",
        "-o", output_template,
        "--no-playlist",
        "--quiet",
        "--progress",
        url
    ]

    try:
        result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            # Try simpler format if merge fails
            print(f"  Retrying with simpler format...")
            download_cmd_simple = [
                "yt-dlp",
                "-f", "best[height<=720]",
                "--download-sections", f"*0-{max_duration}",
                "-o", output_template,
                "--no-playlist",
                "--quiet",
                "--progress",
                url
            ]
            result = subprocess.run(download_cmd_simple, capture_output=True, text=True, timeout=300)

        # Find the downloaded file
        downloaded_files = list(Path(output_dir).glob(f"{safe_title}__{video_id}.*"))
        if downloaded_files:
            filepath = str(downloaded_files[0])
            print(f"  ✓ Downloaded: {filepath}")
            return {
                "success": True,
                "filepath": filepath,
                "title": title,
                "duration": duration,
                "video_id": video_id,
                "skipped": False
            }
        else:
            print(f"  ✗ Download completed but file not found")
            return {"success": False, "error": "file not found after download"}

    except subprocess.TimeoutExpired:
        print(f"  ✗ Download timeout (300s)")
        return {"success": False, "error": "download timeout"}


# ─── Frame Extraction ────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = DEFAULT_FPS,
    max_resolution: int = DEFAULT_MAX_RESOLUTION
) -> dict:
    """
    Extract frames from a video at the specified FPS.

    Returns dict with:
        - frame_count: number of frames extracted
        - frame_dir: directory where frames are saved
        - metadata: video metadata
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        print(f"  ✗ Video file not found: {video_path}")
        return {"frame_count": 0, "error": "file not found"}

    # Create output directory based on video filename
    video_name = Path(video_path).stem
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ Could not open video: {video_path}")
        return {"frame_count": 0, "error": "could not open video"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"  Video: {width}x{height} @ {video_fps:.1f}fps, {duration:.1f}s, {total_frames} frames")

    # Calculate frame interval
    if fps >= video_fps:
        frame_interval = 1  # Extract every frame
    else:
        frame_interval = int(video_fps / fps)

    print(f"  Extracting at {fps} fps (every {frame_interval} frames)...")

    # Calculate resize dimensions
    scale = 1.0
    if max(width, height) > max_resolution:
        scale = max_resolution / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    # Ensure even dimensions (required for some codecs)
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)

    if scale < 1.0:
        print(f"  Resizing: {width}x{height} → {new_width}x{new_height}")

    # Extract frames
    frame_count = 0
    frame_idx = 0
    extracted_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Resize if needed
            if scale < 1.0:
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Calculate timestamp
            timestamp = frame_idx / video_fps

            # Save frame
            frame_filename = f"frame_{frame_count:05d}_t{timestamp:.2f}s.jpg"
            frame_path = os.path.join(frame_dir, frame_filename)
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            extracted_frames.append({
                "filename": frame_filename,
                "timestamp": round(timestamp, 2),
                "frame_index": frame_idx,
                "shape": [new_height, new_width, 3]
            })

            frame_count += 1

        frame_idx += 1

    cap.release()

    # Save metadata
    metadata = {
        "video_path": video_path,
        "video_name": video_name,
        "original_resolution": [width, height],
        "extracted_resolution": [new_width, new_height],
        "video_fps": round(video_fps, 2),
        "extraction_fps": fps,
        "duration_seconds": round(duration, 2),
        "total_video_frames": total_frames,
        "extracted_frame_count": frame_count,
        "frame_interval": frame_interval,
        "frames": extracted_frames,
        "extracted_at": datetime.now().isoformat()
    }

    metadata_path = os.path.join(frame_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Extracted {frame_count} frames → {frame_dir}/")

    return {
        "frame_count": frame_count,
        "frame_dir": frame_dir,
        "metadata": metadata
    }


# ─── Pipeline ────────────────────────────────────────────────────

def process_url(url: str, video_dir: str, frame_dir: str, fps: float,
                max_duration: int = DEFAULT_MAX_DURATION,
                description: str = "") -> dict:
    """Download a video and extract frames. Returns processing results."""
    print(f"\n{'='*60}")
    print(f"Processing: {url}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*60}")

    # Download
    dl_result = download_video(url, video_dir, max_duration)
    if not dl_result.get("success"):
        print(f"  ✗ Download failed: {dl_result.get('error', 'unknown')}")
        return {"success": False, "url": url, "error": dl_result.get("error")}

    # Extract frames
    extract_result = extract_frames(dl_result["filepath"], frame_dir, fps)

    return {
        "success": True,
        "url": url,
        "title": dl_result.get("title", ""),
        "video_path": dl_result["filepath"],
        "frame_dir": extract_result.get("frame_dir", ""),
        "frame_count": extract_result.get("frame_count", 0),
        "duration": dl_result.get("duration", 0)
    }


def process_local(video_path: str, frame_dir: str, fps: float) -> dict:
    """Extract frames from a local video file."""
    print(f"\n{'='*60}")
    print(f"Processing local file: {video_path}")
    print(f"{'='*60}")

    extract_result = extract_frames(video_path, frame_dir, fps)

    return {
        "success": extract_result.get("frame_count", 0) > 0,
        "video_path": video_path,
        "frame_dir": extract_result.get("frame_dir", ""),
        "frame_count": extract_result.get("frame_count", 0)
    }


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Strawberry VLA — Video Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_ingest.py --url "https://youtube.com/watch?v=XXXX" --fps 1
  python video_ingest.py --from-list --fps 0.5
  python video_ingest.py --local my_video.mp4 --fps 2
  python video_ingest.py --from-list --fps 1 --max-duration 300
        """
    )

    # Input sources (mutually exclusive-ish, but we handle gracefully)
    parser.add_argument("--url", type=str, help="YouTube URL to download and process")
    parser.add_argument("--from-list", action="store_true",
                        help="Process all videos from the curated list")
    parser.add_argument("--local", type=str, help="Path to a local video file")

    # Processing options
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help=f"Frames per second to extract (default: {DEFAULT_FPS})")
    parser.add_argument("--max-duration", type=int, default=DEFAULT_MAX_DURATION,
                        help=f"Max video duration in seconds (default: {DEFAULT_MAX_DURATION})")
    parser.add_argument("--max-resolution", type=int, default=DEFAULT_MAX_RESOLUTION,
                        help=f"Max frame resolution (default: {DEFAULT_MAX_RESOLUTION})")

    # Output directories
    parser.add_argument("--video-dir", type=str, default=DEFAULT_VIDEO_DIR,
                        help=f"Directory to save downloaded videos (default: {DEFAULT_VIDEO_DIR})")
    parser.add_argument("--frame-dir", type=str, default=DEFAULT_FRAME_DIR,
                        help=f"Directory to save extracted frames (default: {DEFAULT_FRAME_DIR})")

    # Utility
    parser.add_argument("--list-videos", action="store_true",
                        help="List curated videos and exit")

    args = parser.parse_args()

    # List curated videos
    if args.list_videos:
        print("\nCurated Strawberry Videos:")
        print("-" * 60)
        for i, v in enumerate(CURATED_VIDEOS, 1):
            print(f"  {i}. {v['description']}")
            print(f"     URL: {v['url']}")
            print(f"     Tags: {', '.join(v['tags'])}")
            print()
        return

    # Validate input
    if not args.url and not args.from_list and not args.local:
        parser.print_help()
        print("\nError: Provide --url, --from-list, or --local")
        sys.exit(1)

    print("=" * 60)
    print("Strawberry VLA — Video Ingestion Pipeline")
    print("=" * 60)
    print(f"FPS: {args.fps}")
    print(f"Max duration: {args.max_duration}s")
    print(f"Max resolution: {args.max_resolution}px")
    print(f"Video dir: {args.video_dir}")
    print(f"Frame dir: {args.frame_dir}")

    results = []

    if args.local:
        result = process_local(args.local, args.frame_dir, args.fps)
        results.append(result)

    elif args.url:
        result = process_url(args.url, args.video_dir, args.frame_dir,
                             args.fps, args.max_duration)
        results.append(result)

    elif args.from_list:
        print(f"\nProcessing {len(CURATED_VIDEOS)} curated videos...")
        for video in CURATED_VIDEOS:
            result = process_url(
                video["url"], args.video_dir, args.frame_dir,
                args.fps, args.max_duration, video.get("description", "")
            )
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    total_frames = 0
    for r in results:
        status = "✓" if r.get("success") else "✗"
        frames = r.get("frame_count", 0)
        total_frames += frames
        title = r.get("title", r.get("video_path", "unknown"))[:50]
        print(f"  {status} {title} — {frames} frames")

    print(f"\n  Total: {len(results)} videos, {total_frames} frames")
    print(f"  Frames directory: {os.path.abspath(args.frame_dir)}/")
    print("=" * 60)

    # Save summary
    summary_path = os.path.join(args.frame_dir, "ingestion_summary.json")
    os.makedirs(args.frame_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "fps": args.fps,
                "max_duration": args.max_duration,
                "max_resolution": args.max_resolution
            },
            "results": results
        }, f, indent=2)

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
