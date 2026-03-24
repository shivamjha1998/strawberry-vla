# Strawberry VLA System — Phase 3 Report

**3D Coordinate Estimation & Camera Calibration**

**Date:** March 2026
**Author:** Strawberry VLA Team
**Environment:** Mac Mini M4 (16GB), iPhone 15 Pro (camera)

---

## 1. Executive Summary

Phase 3 extended the Strawberry VLA system with 3D coordinate estimation, enabling spatial localization of detected strawberries for robotic arm integration. Two approaches were implemented:

1. **Monocular 3D estimation** — single-camera depth from known strawberry dimensions
2. **Stereo 3D estimation** — dual-camera disparity-based depth with WLS filtering

A camera calibration pipeline was built using OpenCV checkerboard detection, and the monocular approach was validated with physical measurements using an iPhone 15 Pro.

| Achievement | Detail |
|-------------|--------|
| Monocular depth accuracy | 5–25% error range, median ~10% |
| X-axis accuracy | 0.8–4.0 cm error at 20–39 cm range |
| Calibration pipeline | Checkerboard-based intrinsic + stereo extrinsic |
| Processing speed | YOLO ~10ms + 3D <1ms per frame |
| Integration | 3D available in all tabs (Video, Image, Stereo) |

---

## 2. Implementation

### 2.1 Monocular 3D Estimation

Uses the pinhole camera model to estimate depth from a single image:

```
depth = (focal_length × real_size) / pixel_size
```

**Assumptions:**
- Average strawberry width: 35 mm
- Average strawberry height: 45 mm
- Default horizontal FOV: ~60° (when uncalibrated)

The system computes two independent depth estimates (from width and height), averages them, and assigns a confidence score based on their agreement:

| Agreement | Confidence |
|-----------|------------|
| < 15% difference | High |
| 15–30% difference | Medium |
| > 30% difference | Low |

**3D coordinate system:**
- Origin: camera optical center
- X-axis: right (positive)
- Y-axis: down (positive)
- Z-axis: depth into scene (positive)
- Units: millimeters

### 2.2 Stereo 3D Estimation

For dual-camera setups, the system uses stereo disparity matching with:

- **SGBM** (Semi-Global Block Matching) for disparity computation
- **WLS filter** (Weighted Least Squares) for disparity map refinement
- **Left-right consistency checking** for reliability validation

### 2.3 Camera Calibration Pipeline

Built on OpenCV's `findChessboardCornersSB` with sub-pixel refinement:

1. **Intrinsic calibration** — single camera with checkerboard pattern
2. **Stereo calibration** — paired images for extrinsic parameters (R, T)
3. **Calibration persistence** — save/load via YAML files

The calibration UI in the 3D/Stereo tab accepts:
- Board inner corner dimensions (columns × rows)
- Square size in mm
- Multiple checkerboard images from different angles

---

## 3. Validation

### 3.1 Test Setup

- **Camera:** iPhone 15 Pro (main lens)
- **Subjects:** Real strawberries at measured positions
- **Method:** Single strawberry placed at a known position, additional strawberries added incrementally at measured offsets
- **Mode:** Monocular 3D via Image Upload tab

### 3.2 Depth (Z-axis) Results

| Strawberry | Actual Z | Estimated Z | Error (cm) | Error (%) |
|-----------|----------|-------------|------------|-----------|
| #1 | 30 cm | 27.2 cm | -2.8 | -9.3% |
| #2 | 39 cm | 29.2 cm | -9.8 | -25.1% |
| #3 | 20 cm | 19.0 cm | -1.0 | -5.0% |
| #4 | 36 cm | 32.0 cm | -4.0 | -11.1% |

**Mean absolute error:** 4.4 cm
**Mean percentage error:** -12.6% (consistent underestimation)

### 3.3 Lateral (X-axis) Results

Measured as offset from Strawberry #1 (reference position):

| Strawberry | Actual X | Estimated X | Error (cm) |
|-----------|----------|-------------|------------|
| #2 | -14 cm | -10 cm | 4.0 |
| #3 | 5 cm | 5.8 cm | 0.8 |
| #4 | 4 cm | 5.5 cm | 1.5 |

**Mean absolute error:** 2.1 cm

### 3.4 Analysis

**Depth trends:**
- Closer objects (< 25 cm) are more accurate (5% error)
- Farther objects (> 35 cm) show larger error (11–25%)
- Systematic underestimation across all distances suggests the estimated focal length is slightly low, or the assumed strawberry size is slightly large

**X-axis trends:**
- Direction is always correct
- Magnitude is reasonably accurate (< 2 cm for nearby objects)
- X accuracy is coupled to Z accuracy since X is derived from depth

**Error sources:**
1. **Size variation** — Monocular depth assumes all strawberries are 35×45 mm. Smaller strawberries appear farther, larger ones closer
2. **Focal length estimation** — Without calibration, FOV is assumed to be ~60°. The iPhone 15 Pro main lens has a ~26mm equivalent focal length (~69° FOV), so the default slightly underestimates focal length
3. **Bounding box precision** — YOLO bounding boxes include some background, inflating apparent size and underestimating depth

---

## 4. Architecture Update

### 4.1 New Components

```
strawberry-vla/
├── stereo_calibration.py          # NEW — calibration & 3D estimation
│   ├── CameraCalibrator           #   Single-camera intrinsic calibration
│   ├── StereoCalibrator           #   Stereo pair extrinsic calibration
│   ├── StereoDepthEstimator       #   Dual-camera disparity + depth
│   └── MonocularDepthEstimator    #   Single-camera size-based depth
├── demo_app.py                    # UPDATED — 3D UI in all tabs
│   ├── Video tab                  #   + "3D coordinates" checkbox
│   ├── Image tab                  #   + "3D coordinates" checkbox
│   └── 3D / Stereo tab           #   NEW — calibration + stereo/mono detection
└── strawberry_detector.py         # UPDATED — --mono-3d CLI flag
```

### 4.2 Pipeline Flow

```
Image → YOLO Detection → Bounding Boxes
                              ↓
                    Monocular 3D Estimation
                              ↓
              depth = (f × real_size) / pixel_size
                              ↓
                    X, Y, Z in millimeters
```

---

## 5. Conclusions

1. **Monocular 3D is viable for approximate localization** — median ~10% depth error is sufficient for a robotic arm with adaptive gripping (approach + fine-adjust)
2. **Calibration improves accuracy** — using true camera intrinsics from checkerboard calibration replaces the FOV assumption
3. **Size variation is the dominant error source** — a 35mm assumed width on a 30mm strawberry causes ~17% depth overestimation. This is an inherent limitation of single-camera depth
4. **Stereo would reduce error** — dual-camera depth does not depend on object size assumptions, making it more robust for varying strawberry sizes

### Recommended Next Steps

- **Use calibrated focal length** for monocular estimates (load saved calibration automatically)
- **Stereo camera hardware** for production accuracy (removes size assumption dependency)
- **Close-range sensor on gripper** for final positioning (ultrasonic or ToF)
- **Per-variety size profiles** to improve monocular accuracy for known cultivars