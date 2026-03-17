"""
stereo_calibration.py — Stereo Vision & 3D Coordinate System

Multi-camera calibration and 3D triangulation for strawberry localization.
Builds on the existing YOLO detection pipeline to add depth estimation
using stereo vision (two synchronized cameras).

Workflow:
    1. Calibrate each camera individually (checkerboard intrinsics)
    2. Calibrate stereo pair (extrinsic R, T between cameras)
    3. Rectify stereo images for efficient stereo matching
    4. Compute disparity → depth → 3D coordinates per detection

Usage:
    # Standalone demo with synthetic stereo
    python stereo_calibration.py --image frame.jpg

    # Calibrate from checkerboard images
    python stereo_calibration.py --calibrate --left-dir calib/left/ --right-dir calib/right/

    # Process stereo pair with existing calibration
    python stereo_calibration.py --left img_L.jpg --right img_R.jpg --calibration cal.yaml
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np


# ─── Camera Intrinsic Calibration ────────────────────────────────────────────


class CameraCalibrator:
    """Single-camera intrinsic calibration using checkerboard patterns."""

    def __init__(self, board_size=(9, 6), square_size_mm=25.0):
        """
        Args:
            board_size: (cols, rows) inner corners of the checkerboard.
            square_size_mm: Physical size of each square in millimetres.
        """
        self.board_size = board_size
        self.square_size_mm = square_size_mm
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None  # (width, height)
        self.rms_error = None
        self._obj_points = []
        self._img_points = []
        self._calibration_images = []

    # ── Corner detection ──

    def find_corners(self, image_path):
        """Find checkerboard corners in an image.

        Returns:
            (found, corners, vis_image) — corners is Nx1x2 float32 or None,
            vis_image is a copy with corners drawn (or the original if not found).
        """
        img = cv2.imread(image_path)
        if img is None:
            return False, None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()

        # Try the newer SB variant first (more robust), fall back to classic
        found = False
        corners = None
        try:
            found, corners = cv2.findChessboardCornersSB(gray, self.board_size)
        except AttributeError:
            pass

        if not found:
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                     | cv2.CALIB_CB_NORMALIZE_IMAGE
                     | cv2.CALIB_CB_FAST_CHECK)
            found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)

        if found and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, self.board_size, corners, found)

        return found, corners, vis

    def add_calibration_image(self, image_path):
        """Add a checkerboard image for calibration.

        Returns True if corners were found and added.
        """
        found, corners, _ = self.find_corners(image_path)
        if not found:
            return False

        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        if self.image_size is None:
            self.image_size = (w, h)

        # Build 3-D object points for the checkerboard
        obj_pts = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        obj_pts[:, :2] = (
            np.mgrid[0:self.board_size[0], 0:self.board_size[1]]
            .T.reshape(-1, 2) * self.square_size_mm
        )

        self._obj_points.append(obj_pts)
        self._img_points.append(corners)
        self._calibration_images.append(image_path)
        return True

    # ── Calibration ──

    def calibrate(self, min_images=5):
        """Run intrinsic calibration.

        Returns:
            Summary dict with rms_error, focal_length_px, principal_point, etc.
        Raises:
            ValueError: If fewer than min_images have been added.
        """
        if len(self._obj_points) < min_images:
            raise ValueError(
                f"Need at least {min_images} calibration images, "
                f"got {len(self._obj_points)}"
            )

        rms, mtx, dist, _, _ = cv2.calibrateCamera(
            self._obj_points, self._img_points, self.image_size, None, None,
        )
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rms_error = rms

        return {
            "rms_error": round(rms, 4),
            "num_images": len(self._obj_points),
            "image_size": self.image_size,
            "focal_length_px": (round(mtx[0, 0], 2), round(mtx[1, 1], 2)),
            "principal_point": (round(mtx[0, 2], 2), round(mtx[1, 2], 2)),
        }

    # ── Undistortion ──

    def undistort_image(self, image):
        """Remove lens distortion from an image.

        Args:
            image: BGR numpy array.
        Returns:
            Undistorted BGR numpy array.
        """
        if not self.is_calibrated:
            raise RuntimeError("Camera not calibrated yet.")
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    # ── Persistence ──

    def save(self, filepath):
        """Save calibration to YAML via cv2.FileStorage."""
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        fs.write("board_cols", self.board_size[0])
        fs.write("board_rows", self.board_size[1])
        fs.write("square_size_mm", self.square_size_mm)
        fs.write("image_width", self.image_size[0])
        fs.write("image_height", self.image_size[1])
        fs.write("camera_matrix", self.camera_matrix)
        fs.write("dist_coeffs", self.dist_coeffs)
        fs.write("rms_error", self.rms_error)
        fs.release()

    def load(self, filepath):
        """Load calibration from YAML."""
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        self.board_size = (
            int(fs.getNode("board_cols").real()),
            int(fs.getNode("board_rows").real()),
        )
        self.square_size_mm = fs.getNode("square_size_mm").real()
        self.image_size = (
            int(fs.getNode("image_width").real()),
            int(fs.getNode("image_height").real()),
        )
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("dist_coeffs").mat()
        self.rms_error = fs.getNode("rms_error").real()
        fs.release()

    @property
    def is_calibrated(self):
        return self.camera_matrix is not None


# ─── Stereo Pair Calibration ─────────────────────────────────────────────────


class StereoCalibrator:
    """Stereo camera pair calibration — extrinsic parameters + rectification."""

    def __init__(self, cam_left, cam_right):
        """
        Args:
            cam_left: CameraCalibrator for the left camera.
            cam_right: CameraCalibrator for the right camera.
        """
        self.cam_left = cam_left
        self.cam_right = cam_right

        # Extrinsic parameters
        self.R = None   # 3x3 rotation
        self.T = None   # 3x1 translation
        self.E = None   # essential matrix
        self.F = None   # fundamental matrix
        self.rms_error = None

        # Rectification
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None   # 4x4 disparity-to-depth
        self._map_left = None   # (map1, map2) remap LUTs
        self._map_right = None
        self._valid_roi_left = None
        self._valid_roi_right = None

        # Accumulated stereo corner data
        self._stereo_obj_points = []
        self._stereo_points_left = []
        self._stereo_points_right = []

    def add_stereo_pair(self, left_path, right_path):
        """Add a synchronised checkerboard image pair.

        Returns True if corners were found in both images.
        """
        found_l, corners_l, _ = self.cam_left.find_corners(left_path)
        found_r, corners_r, _ = self.cam_right.find_corners(right_path)

        if not (found_l and found_r):
            return False

        board = self.cam_left.board_size
        obj_pts = np.zeros((board[0] * board[1], 3), np.float32)
        obj_pts[:, :2] = (
            np.mgrid[0:board[0], 0:board[1]]
            .T.reshape(-1, 2) * self.cam_left.square_size_mm
        )

        self._stereo_obj_points.append(obj_pts)
        self._stereo_points_left.append(corners_l)
        self._stereo_points_right.append(corners_r)
        return True

    def calibrate_stereo(self, min_pairs=5):
        """Run stereo calibration using pre-calibrated individual cameras.

        Returns:
            Summary dict with rms_error, baseline_mm, etc.
        Raises:
            RuntimeError: If individual cameras are not calibrated.
            ValueError: If fewer than min_pairs stereo pairs available.
        """
        if not (self.cam_left.is_calibrated and self.cam_right.is_calibrated):
            raise RuntimeError(
                "Both cameras must be individually calibrated first."
            )
        if len(self._stereo_obj_points) < min_pairs:
            raise ValueError(
                f"Need at least {min_pairs} stereo pairs, "
                f"got {len(self._stereo_obj_points)}"
            )

        img_size = self.cam_left.image_size

        rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            self._stereo_obj_points,
            self._stereo_points_left,
            self._stereo_points_right,
            self.cam_left.camera_matrix,
            self.cam_left.dist_coeffs,
            self.cam_right.camera_matrix,
            self.cam_right.dist_coeffs,
            img_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )

        self.R, self.T, self.E, self.F = R, T, E, F
        self.rms_error = rms

        # Rectification
        self.R1, self.R2, self.P1, self.P2, self.Q, roi_l, roi_r = cv2.stereoRectify(
            self.cam_left.camera_matrix, self.cam_left.dist_coeffs,
            self.cam_right.camera_matrix, self.cam_right.dist_coeffs,
            img_size, R, T,
            alpha=0,  # crop to valid pixels only
        )
        self._valid_roi_left = roi_l
        self._valid_roi_right = roi_r

        # Build undistort + rectify maps
        self._map_left = cv2.initUndistortRectifyMap(
            self.cam_left.camera_matrix, self.cam_left.dist_coeffs,
            self.R1, self.P1, img_size, cv2.CV_32FC1,
        )
        self._map_right = cv2.initUndistortRectifyMap(
            self.cam_right.camera_matrix, self.cam_right.dist_coeffs,
            self.R2, self.P2, img_size, cv2.CV_32FC1,
        )

        baseline_mm = float(np.linalg.norm(T))
        return {
            "rms_error": round(rms, 4),
            "baseline_mm": round(baseline_mm, 2),
            "num_pairs": len(self._stereo_obj_points),
            "image_size": img_size,
        }

    def rectify_pair(self, left_img, right_img):
        """Apply rectification to a stereo image pair.

        Returns:
            (left_rectified, right_rectified) — BGR numpy arrays.
        """
        if not self.is_calibrated:
            raise RuntimeError("Stereo pair not calibrated yet.")
        left_rect = cv2.remap(left_img, *self._map_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, *self._map_right, cv2.INTER_LINEAR)
        return left_rect, right_rect

    # ── Persistence ──

    def save(self, filepath):
        """Save full stereo calibration (intrinsics + extrinsics + rectification)."""
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        # Left camera intrinsics
        fs.write("left_camera_matrix", self.cam_left.camera_matrix)
        fs.write("left_dist_coeffs", self.cam_left.dist_coeffs)
        fs.write("left_rms_error", self.cam_left.rms_error if self.cam_left.rms_error else 0.0)
        # Right camera intrinsics
        fs.write("right_camera_matrix", self.cam_right.camera_matrix)
        fs.write("right_dist_coeffs", self.cam_right.dist_coeffs)
        fs.write("right_rms_error", self.cam_right.rms_error if self.cam_right.rms_error else 0.0)
        # Common
        fs.write("image_width", self.cam_left.image_size[0])
        fs.write("image_height", self.cam_left.image_size[1])
        fs.write("board_cols", self.cam_left.board_size[0])
        fs.write("board_rows", self.cam_left.board_size[1])
        fs.write("square_size_mm", self.cam_left.square_size_mm)
        # Stereo extrinsics
        fs.write("R", self.R)
        fs.write("T", self.T)
        fs.write("E", self.E)
        fs.write("F", self.F)
        fs.write("stereo_rms_error", self.rms_error)
        # Rectification
        fs.write("R1", self.R1)
        fs.write("R2", self.R2)
        fs.write("P1", self.P1)
        fs.write("P2", self.P2)
        fs.write("Q", self.Q)
        fs.release()

    def load(self, filepath):
        """Load full stereo calibration from YAML."""
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        # Left camera
        self.cam_left.camera_matrix = fs.getNode("left_camera_matrix").mat()
        self.cam_left.dist_coeffs = fs.getNode("left_dist_coeffs").mat()
        self.cam_left.rms_error = fs.getNode("left_rms_error").real()
        # Right camera
        self.cam_right.camera_matrix = fs.getNode("right_camera_matrix").mat()
        self.cam_right.dist_coeffs = fs.getNode("right_dist_coeffs").mat()
        self.cam_right.rms_error = fs.getNode("right_rms_error").real()
        # Common
        w = int(fs.getNode("image_width").real())
        h = int(fs.getNode("image_height").real())
        self.cam_left.image_size = (w, h)
        self.cam_right.image_size = (w, h)
        self.cam_left.board_size = (
            int(fs.getNode("board_cols").real()),
            int(fs.getNode("board_rows").real()),
        )
        self.cam_right.board_size = self.cam_left.board_size
        self.cam_left.square_size_mm = fs.getNode("square_size_mm").real()
        self.cam_right.square_size_mm = self.cam_left.square_size_mm
        # Stereo extrinsics
        self.R = fs.getNode("R").mat()
        self.T = fs.getNode("T").mat()
        self.E = fs.getNode("E").mat()
        self.F = fs.getNode("F").mat()
        self.rms_error = fs.getNode("stereo_rms_error").real()
        # Rectification
        self.R1 = fs.getNode("R1").mat()
        self.R2 = fs.getNode("R2").mat()
        self.P1 = fs.getNode("P1").mat()
        self.P2 = fs.getNode("P2").mat()
        self.Q = fs.getNode("Q").mat()
        # Rebuild remap LUTs
        img_size = (w, h)
        self._map_left = cv2.initUndistortRectifyMap(
            self.cam_left.camera_matrix, self.cam_left.dist_coeffs,
            self.R1, self.P1, img_size, cv2.CV_32FC1,
        )
        self._map_right = cv2.initUndistortRectifyMap(
            self.cam_right.camera_matrix, self.cam_right.dist_coeffs,
            self.R2, self.P2, img_size, cv2.CV_32FC1,
        )
        fs.release()

    @property
    def is_calibrated(self):
        return self.R is not None and self._map_left is not None


# ─── Stereo Depth Estimation ─────────────────────────────────────────────────


class StereoDepthEstimator:
    """Compute depth maps and 3D coordinates from calibrated stereo pairs.

    Uses StereoSGBM with left-right consistency checking and optional
    WLS (Weighted Least Squares) disparity filtering for improved quality
    in textureless regions and near depth discontinuities.
    """

    # Check for cv2.ximgproc (WLS filter) at class level
    _has_wls = hasattr(cv2, "ximgproc")

    def __init__(self, stereo_calibrator):
        """
        Args:
            stereo_calibrator: A calibrated StereoCalibrator instance.
        """
        self.stereo_cal = stereo_calibrator

        # StereoSGBM parameters tuned for greenhouse 0.3–1.5 m
        self.min_disparity = 0
        self.num_disparities = 128   # must be divisible by 16
        self.block_size = 5
        self.P1 = 8 * 3 * self.block_size ** 2
        self.P2 = 32 * 3 * self.block_size ** 2
        self.disp12_max_diff = 1
        self.uniqueness_ratio = 10
        self.speckle_window_size = 100
        self.speckle_range = 32
        self.pre_filter_cap = 63

        # WLS filter parameters (used when cv2.ximgproc is available)
        self.wls_lambda = 8000.0
        self.wls_sigma = 1.5

    def _create_matcher(self):
        """Create a StereoSGBM matcher with current parameters."""
        return cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.P1,
            P2=self.P2,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            preFilterCap=self.pre_filter_cap,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def _create_right_matcher(self, left_matcher):
        """Create a right-to-left matcher for consistency checking."""
        if self._has_wls:
            return cv2.ximgproc.createRightMatcher(left_matcher)
        # Manual right matcher with flipped disparity range
        return cv2.StereoSGBM_create(
            minDisparity=-self.num_disparities,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.P1,
            P2=self.P2,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            preFilterCap=self.pre_filter_cap,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute_disparity(self, left_rect, right_rect):
        """Compute disparity map from a rectified stereo pair.

        Uses left-right consistency checking and WLS filtering (if cv2.ximgproc
        is available) to improve quality. Falls back to bilateral filtering
        when ximgproc is not installed.

        Args:
            left_rect: Rectified left image (BGR).
            right_rect: Rectified right image (BGR).
        Returns:
            Float32 disparity map (pixels). Invalid regions are set to 0.
        """
        gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        left_matcher = self._create_matcher()
        disp_left = left_matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # Left-right consistency: compute right-to-left disparity
        right_matcher = self._create_right_matcher(left_matcher)
        disp_right = right_matcher.compute(gray_r, gray_l).astype(np.float32) / 16.0

        if self._has_wls:
            # WLS filter: fills holes, preserves edges, uses L-R consistency
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls_filter.setLambda(self.wls_lambda)
            wls_filter.setSigmaColor(self.wls_sigma)
            disparity = wls_filter.filter(
                (disp_left * 16).astype(np.int16), left_rect,
                disparity_map_right=(disp_right * 16).astype(np.int16),
            ).astype(np.float32) / 16.0
        else:
            # Fallback: left-right consistency mask + bilateral filter
            # Mark pixels as invalid where L-R disparity disagrees by > 1 px
            h, w = disp_left.shape
            lr_mask = np.ones((h, w), dtype=bool)
            for y in range(h):
                for x in range(w):
                    d = disp_left[y, x]
                    if d <= 0:
                        lr_mask[y, x] = False
                        continue
                    xr = int(round(x - d))
                    if 0 <= xr < w:
                        if abs(d + disp_right[y, xr]) > 1.0:
                            lr_mask[y, x] = False
                    else:
                        lr_mask[y, x] = False

            disparity = disp_left.copy()
            disparity[~lr_mask] = 0

            # Bilateral filter on valid disparity to smooth while preserving edges
            disp_8u = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_filtered = cv2.bilateralFilter(disp_8u, 9, 75, 75).astype(np.float32)
            # Rescale back and keep only originally valid regions
            if disparity.max() > 0:
                disp_filtered = disp_filtered * (disparity.max() / 255.0)
            disp_filtered[~lr_mask] = 0
            disparity = disp_filtered

        # Final cleanup
        disparity[disparity <= 0] = 0
        return disparity

    def disparity_to_depth(self, disparity):
        """Convert disparity map to depth map using the Q matrix.

        Returns:
            3-channel float32 array from cv2.reprojectImageTo3D (X, Y, Z in mm).
        """
        if self.stereo_cal.Q is None:
            raise RuntimeError("No Q matrix — stereo calibration required.")
        return cv2.reprojectImageTo3D(disparity, self.stereo_cal.Q, handleMissingValues=True)

    def get_3d_point(self, disparity, pixel_x, pixel_y, window_size=5):
        """Get 3D coordinates for a single pixel using a local disparity average.

        Returns:
            Dict with x_mm, y_mm, z_mm, depth_mm, confidence — or None if invalid.
        """
        h, w = disparity.shape[:2]
        half = window_size // 2
        y1 = max(0, pixel_y - half)
        y2 = min(h, pixel_y + half + 1)
        x1 = max(0, pixel_x - half)
        x2 = min(w, pixel_x + half + 1)

        patch = disparity[y1:y2, x1:x2]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None

        disp_std = float(np.std(valid))

        # Reproject using the mean disparity and Q matrix
        points_3d = self.disparity_to_depth(disparity)

        # Use the mean disparity region rather than the exact pixel
        # to avoid sentinel values at invalid pixels
        patch_3d = points_3d[y1:y2, x1:x2]
        valid_z = patch_3d[:, :, 2]
        valid_mask = (valid_z > 0) & (valid_z < 9999)
        if not np.any(valid_mask):
            return None

        x_mm = float(np.mean(patch_3d[:, :, 0][valid_mask]))
        y_mm = float(np.mean(patch_3d[:, :, 1][valid_mask]))
        z_mm = float(np.mean(valid_z[valid_mask]))

        # Sanity: reject nonsensical depths
        if z_mm <= 0 or z_mm >= 9999 or np.isnan(z_mm):
            return None

        # Confidence based on disparity variance
        if disp_std < 1.0:
            confidence = "high"
        elif disp_std < 3.0:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "x_mm": round(x_mm, 1),
            "y_mm": round(y_mm, 1),
            "z_mm": round(z_mm, 1),
            "depth_mm": round(z_mm, 1),
            "confidence": confidence,
        }

    def get_3d_for_bbox(self, disparity, bbox):
        """Compute 3D position and size for a 2D bounding box.

        Uses the central region of the bbox (inner 50%) for depth estimation
        to avoid edge artifacts. Confidence is based on disparity coverage
        and variance within the bbox.

        Args:
            disparity: Float32 disparity map (from compute_disparity).
            bbox: [x1, y1, x2, y2] in rectified left image coordinates.
        Returns:
            Dict with center_3d, depth_mm, size_3d, confidence — or None.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w_box, h_box = x2 - x1, y2 - y1
        if w_box <= 0 or h_box <= 0:
            return None

        # Use inner 50% of bbox to avoid edge noise
        margin_x, margin_y = w_box // 4, h_box // 4
        inner_x1 = x1 + margin_x
        inner_y1 = y1 + margin_y
        inner_x2 = x2 - margin_x
        inner_y2 = y2 - margin_y

        cx = (inner_x1 + inner_x2) // 2
        cy = (inner_y1 + inner_y2) // 2
        win = max(5, min(inner_x2 - inner_x1, inner_y2 - inner_y1))

        pt = self.get_3d_point(disparity, cx, cy, window_size=win)
        if pt is None:
            return None

        # Compute disparity coverage within the full bbox for confidence
        h, w = disparity.shape[:2]
        bx1, by1 = max(0, x1), max(0, y1)
        bx2, by2 = min(w, x2), min(h, y2)
        bbox_disp = disparity[by1:by2, bx1:bx2]
        total_px = bbox_disp.size
        valid_px = np.count_nonzero(bbox_disp > 0)
        coverage = valid_px / max(1, total_px)

        # Upgrade/downgrade confidence based on coverage
        confidence = pt["confidence"]
        if coverage < 0.2:
            confidence = "low"
        elif coverage < 0.5 and confidence == "high":
            confidence = "medium"

        # Estimate 3D size from bbox pixel dimensions and depth
        depth_mm = pt["z_mm"]
        if self.stereo_cal.P1 is not None:
            fx = self.stereo_cal.P1[0, 0]
        else:
            fx = self.stereo_cal.cam_left.camera_matrix[0, 0]

        width_mm = (w_box * depth_mm) / fx if fx > 0 else 0
        height_mm = (h_box * depth_mm) / fx if fx > 0 else 0

        return {
            "center_3d": {
                "x_mm": pt["x_mm"],
                "y_mm": pt["y_mm"],
                "z_mm": pt["z_mm"],
            },
            "depth_mm": pt["depth_mm"],
            "size_3d": {
                "width_mm": round(width_mm, 1),
                "height_mm": round(height_mm, 1),
            },
            "confidence": confidence,
            "disparity_coverage": round(coverage, 2),
        }

    def compute_full_3d(self, left_rect, right_rect, detections):
        """Compute disparity and add 3D coordinates to each detection.

        Args:
            left_rect: Rectified left image (BGR).
            right_rect: Rectified right image (BGR).
            detections: List of detection dicts with "bbox_2d" keys.
        Returns:
            (disparity_map, detections_with_3d) — each detection gains "position_3d".
        """
        disparity = self.compute_disparity(left_rect, right_rect)

        for det in detections:
            bbox = det.get("bbox_2d", [])
            if len(bbox) != 4:
                continue
            pos_3d = self.get_3d_for_bbox(disparity, bbox)
            if pos_3d is not None:
                det["position_3d"] = pos_3d

        return disparity, detections


# ─── Monocular Depth Estimation ──────────────────────────────────────────────


# Average strawberry dimensions (mm) from agricultural research
STRAWBERRY_WIDTH_MM = 35.0   # typical width at widest point
STRAWBERRY_HEIGHT_MM = 45.0  # typical length including calyx


class MonocularDepthEstimator:
    """Estimate 3D coordinates from a single camera using known object size.

    Uses the pinhole camera model: depth = (focal_length * real_size) / pixel_size.
    Requires camera intrinsics (focal length) and assumes average strawberry
    dimensions. Accuracy depends on how close each strawberry is to the assumed
    average size, typically +-20% for mature strawberries.

    This approach is based on the PnP (Perspective-n-Point) principle used in
    recent single-shot 6DoF pose estimation papers for robotic harvesting.
    """

    def __init__(self, focal_length_px=None, camera_matrix=None,
                 image_size=(640, 480),
                 real_width_mm=STRAWBERRY_WIDTH_MM,
                 real_height_mm=STRAWBERRY_HEIGHT_MM):
        """
        Args:
            focal_length_px: Focal length in pixels. If None, estimated from
                image width assuming ~60 degree horizontal FOV.
            camera_matrix: 3x3 intrinsic matrix (overrides focal_length_px).
            image_size: (width, height) of the images.
            real_width_mm: Expected real-world width of a strawberry (mm).
            real_height_mm: Expected real-world height of a strawberry (mm).
        """
        self.real_width_mm = real_width_mm
        self.real_height_mm = real_height_mm
        self.image_size = image_size

        if camera_matrix is not None:
            self.fx = float(camera_matrix[0, 0])
            self.fy = float(camera_matrix[1, 1])
            self.cx = float(camera_matrix[0, 2])
            self.cy = float(camera_matrix[1, 2])
        elif focal_length_px is not None:
            self.fx = self.fy = float(focal_length_px)
            self.cx = image_size[0] / 2.0
            self.cy = image_size[1] / 2.0
        else:
            # Estimate focal length assuming ~60 degree horizontal FOV
            self.fx = self.fy = image_size[0] / (2.0 * np.tan(np.radians(30)))
            self.cx = image_size[0] / 2.0
            self.cy = image_size[1] / 2.0

    def estimate_depth_from_bbox(self, bbox):
        """Estimate depth (Z) from a 2D bounding box using known object size.

        Uses both width and height estimates and averages them for robustness.

        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates.
        Returns:
            Dict with depth_mm, confidence, method — or None if bbox is invalid.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w_px = x2 - x1
        h_px = y2 - y1
        if w_px <= 0 or h_px <= 0:
            return None

        # Depth from width: Z = (fx * real_width) / pixel_width
        z_from_width = (self.fx * self.real_width_mm) / w_px
        # Depth from height: Z = (fy * real_height) / pixel_height
        z_from_height = (self.fy * self.real_height_mm) / h_px

        # Average both estimates (more robust than either alone)
        z_mm = (z_from_width + z_from_height) / 2.0

        # Confidence based on agreement between width and height estimates
        ratio = min(z_from_width, z_from_height) / max(z_from_width, z_from_height)
        if ratio > 0.85:
            confidence = "high"
        elif ratio > 0.65:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "depth_mm": round(z_mm, 1),
            "depth_from_width_mm": round(z_from_width, 1),
            "depth_from_height_mm": round(z_from_height, 1),
            "confidence": confidence,
        }

    def get_3d_for_bbox(self, bbox):
        """Compute full 3D position for a detection bounding box.

        Projects the bbox centre to 3D using estimated depth.

        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates.
        Returns:
            Dict with center_3d, depth_mm, size_3d, confidence — or None.
        """
        depth_info = self.estimate_depth_from_bbox(bbox)
        if depth_info is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        z_mm = depth_info["depth_mm"]

        # Back-project pixel centre to 3D camera coordinates
        x_mm = (cx_px - self.cx) * z_mm / self.fx
        y_mm = (cy_px - self.cy) * z_mm / self.fy

        return {
            "center_3d": {
                "x_mm": round(x_mm, 1),
                "y_mm": round(y_mm, 1),
                "z_mm": round(z_mm, 1),
            },
            "depth_mm": round(z_mm, 1),
            "size_3d": {
                "width_mm": round(self.real_width_mm, 1),
                "height_mm": round(self.real_height_mm, 1),
            },
            "confidence": depth_info["confidence"],
            "method": "monocular_size",
        }

    def compute_full_3d(self, detections):
        """Add 3D coordinates to each detection using monocular depth.

        Args:
            detections: List of detection dicts with "bbox_2d" keys.
        Returns:
            List of detections with "position_3d" added where possible.
        """
        for det in detections:
            bbox = det.get("bbox_2d", [])
            if len(bbox) != 4:
                continue
            pos_3d = self.get_3d_for_bbox(bbox)
            if pos_3d is not None:
                det["position_3d"] = pos_3d

        return detections


# ─── Synthetic / Demo Utilities ──────────────────────────────────────────────


def generate_synthetic_stereo_pair(image_path, baseline_px=40, depth_variation=True):
    """Create a simulated stereo pair from a single image.

    Shifts the image horizontally to mimic a second camera view.
    For demo/testing when no stereo hardware is available.

    Args:
        image_path: Path to source image.
        baseline_px: Horizontal shift in pixels (simulated disparity).
        depth_variation: If True, add slight perspective warp for realism.
    Returns:
        (left_image, right_image) — BGR numpy arrays.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    left = img.copy()

    if depth_variation:
        # Slight perspective warp to simulate parallax
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([
            [baseline_px, 2],
            [w + baseline_px - 5, -2],
            [w + baseline_px - 3, h + 2],
            [baseline_px + 2, h - 2],
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        right = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    else:
        # Simple horizontal shift
        M = np.float32([[1, 0, baseline_px], [0, 1, 0]])
        right = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return left, right


def generate_synthetic_calibration(image_size=(640, 480), focal_length_px=800.0,
                                    baseline_mm=60.0):
    """Create a StereoCalibrator with realistic synthetic parameters.

    Simulates typical USB/industrial cameras at ~640x480, 4mm lens.
    For demo/testing without real calibration data.

    Args:
        image_size: (width, height) of the images.
        focal_length_px: Focal length in pixels.
        baseline_mm: Stereo baseline in millimetres.
    Returns:
        A fully-populated StereoCalibrator ready for use.
    """
    w, h = image_size
    fx = fy = focal_length_px
    cx, cy = w / 2.0, h / 2.0

    cam_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    dist = np.zeros((1, 5), dtype=np.float64)

    cam_left = CameraCalibrator()
    cam_left.camera_matrix = cam_matrix.copy()
    cam_left.dist_coeffs = dist.copy()
    cam_left.image_size = image_size
    cam_left.rms_error = 0.0

    cam_right = CameraCalibrator()
    cam_right.camera_matrix = cam_matrix.copy()
    cam_right.dist_coeffs = dist.copy()
    cam_right.image_size = image_size
    cam_right.rms_error = 0.0

    stereo = StereoCalibrator(cam_left, cam_right)

    # Identity rotation (parallel cameras)
    stereo.R = np.eye(3, dtype=np.float64)
    # Translation: purely along X-axis
    stereo.T = np.array([[-baseline_mm], [0], [0]], dtype=np.float64)
    stereo.E = np.zeros((3, 3), dtype=np.float64)
    stereo.F = np.zeros((3, 3), dtype=np.float64)
    stereo.rms_error = 0.0

    # For parallel cameras, rectification is identity
    stereo.R1 = np.eye(3, dtype=np.float64)
    stereo.R2 = np.eye(3, dtype=np.float64)
    stereo.P1 = np.hstack([cam_matrix, np.zeros((3, 1))]).astype(np.float64)
    stereo.P2 = np.hstack([cam_matrix, np.array([[baseline_mm * fx], [0], [0]])]).astype(np.float64)

    # Q matrix for disparity → 3D reprojection
    # Q = [[1, 0, 0, -cx],
    #      [0, 1, 0, -cy],
    #      [0, 0, 0,  fx],
    #      [0, 0, -1/Tx, 0]]
    stereo.Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0,  fx],
        [0, 0, -1.0 / baseline_mm, 0],
    ], dtype=np.float64)

    # Build identity remap LUTs (no distortion to correct)
    stereo._map_left = cv2.initUndistortRectifyMap(
        cam_matrix, dist, stereo.R1, stereo.P1, image_size, cv2.CV_32FC1,
    )
    stereo._map_right = cv2.initUndistortRectifyMap(
        cam_matrix, dist, stereo.R2, stereo.P2, image_size, cv2.CV_32FC1,
    )

    return stereo


# ─── Visualisation Utilities ─────────────────────────────────────────────────


def draw_depth_overlay(image, disparity, detections=None, colormap=cv2.COLORMAP_JET,
                       alpha=0.4):
    """Overlay colour-coded depth on an image with optional 3D labels.

    Args:
        image: BGR image (left rectified).
        disparity: Float32 disparity map.
        detections: Optional list of dicts with "position_3d" keys.
        colormap: OpenCV colormap constant.
        alpha: Blend factor for overlay.
    Returns:
        BGR image with depth overlay and labels.
    """
    # Normalise disparity for visualisation
    valid = disparity[disparity > 0]
    if len(valid) == 0:
        return image.copy()

    d_min, d_max = float(valid.min()), float(valid.max())
    if d_max <= d_min:
        d_max = d_min + 1

    norm = ((disparity - d_min) / (d_max - d_min) * 255).clip(0, 255).astype(np.uint8)
    norm[disparity <= 0] = 0
    colour = cv2.applyColorMap(norm, colormap)

    # Blend
    vis = image.copy()
    mask = disparity > 0
    vis[mask] = cv2.addWeighted(image, 1 - alpha, colour, alpha, 0)[mask]

    # Draw 3D labels on detections
    if detections:
        for i, det in enumerate(detections):
            pos = det.get("position_3d")
            if pos is None:
                continue
            bbox = det.get("bbox_2d", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            depth_cm = pos["depth_mm"] / 10.0
            conf = pos["confidence"]
            label = f"#{i+1} z:{depth_cm:.0f}cm ({conf})"

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y2), (x1 + tw + 4, y2 + th + 8), (0, 0, 0), -1)
            cv2.putText(vis, label, (x1 + 2, y2 + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    return vis


def draw_epipolar_lines(left_rect, right_rect, num_lines=20):
    """Draw horizontal lines on a side-by-side rectified pair for QA.

    Corresponding features should lie on the same horizontal line.

    Returns:
        BGR image showing both views side-by-side with coloured horizontal lines.
    """
    h, w = left_rect.shape[:2]
    canvas = np.hstack([left_rect.copy(), right_rect.copy()])

    step = max(1, h // (num_lines + 1))
    for i in range(1, num_lines + 1):
        y = i * step
        colour = tuple(int(c) for c in np.random.randint(0, 255, 3).tolist())
        cv2.line(canvas, (0, y), (w * 2, y), colour, 1)

    return canvas


# ─── Main (standalone testing) ───────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stereo Calibration & 3D Coordinate System",
    )
    parser.add_argument("--image", type=str, help="Single image for synthetic stereo demo")
    parser.add_argument("--left", type=str, help="Left stereo image")
    parser.add_argument("--right", type=str, help="Right stereo image")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration workflow")
    parser.add_argument("--left-dir", type=str, help="Directory of left checkerboard images")
    parser.add_argument("--right-dir", type=str, help="Directory of right checkerboard images")
    parser.add_argument("--calibration", type=str, help="Path to calibration YAML")
    parser.add_argument("--board-cols", type=int, default=9)
    parser.add_argument("--board-rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=25.0)
    parser.add_argument("--output", type=str, default="stereo_output")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Mode 1: Calibration from checkerboard images ──
    if args.calibrate and args.left_dir and args.right_dir:
        print("=" * 50)
        print("  Stereo Calibration")
        print("=" * 50)

        cam_left = CameraCalibrator(
            board_size=(args.board_cols, args.board_rows),
            square_size_mm=args.square_size,
        )
        cam_right = CameraCalibrator(
            board_size=(args.board_cols, args.board_rows),
            square_size_mm=args.square_size,
        )

        left_imgs = sorted(Path(args.left_dir).glob("*.jpg")) + sorted(Path(args.left_dir).glob("*.png"))
        right_imgs = sorted(Path(args.right_dir).glob("*.jpg")) + sorted(Path(args.right_dir).glob("*.png"))

        if len(left_imgs) != len(right_imgs):
            print(f"  Warning: {len(left_imgs)} left vs {len(right_imgs)} right images")

        # Add individual camera images
        print(f"\n  Adding calibration images...")
        for img_path in left_imgs:
            ok = cam_left.add_calibration_image(str(img_path))
            print(f"    Left  {img_path.name}: {'OK' if ok else 'SKIP (no corners)'}")
        for img_path in right_imgs:
            ok = cam_right.add_calibration_image(str(img_path))
            print(f"    Right {img_path.name}: {'OK' if ok else 'SKIP (no corners)'}")

        # Calibrate individually
        print("\n  Calibrating left camera...")
        left_info = cam_left.calibrate(min_images=3)
        print(f"    RMS error: {left_info['rms_error']}")
        print(f"    Focal length: {left_info['focal_length_px']}")

        print("\n  Calibrating right camera...")
        right_info = cam_right.calibrate(min_images=3)
        print(f"    RMS error: {right_info['rms_error']}")

        # Add stereo pairs
        stereo = StereoCalibrator(cam_left, cam_right)
        print(f"\n  Adding stereo pairs...")
        pairs = min(len(left_imgs), len(right_imgs))
        for i in range(pairs):
            ok = stereo.add_stereo_pair(str(left_imgs[i]), str(right_imgs[i]))
            print(f"    Pair {i+1}: {'OK' if ok else 'SKIP'}")

        # Stereo calibration
        print("\n  Running stereo calibration...")
        stereo_info = stereo.calibrate_stereo(min_pairs=3)
        print(f"    Stereo RMS: {stereo_info['rms_error']}")
        print(f"    Baseline: {stereo_info['baseline_mm']} mm")

        # Save
        cal_path = os.path.join(args.output, "stereo_calibration.yaml")
        stereo.save(cal_path)
        print(f"\n  Saved calibration to {cal_path}")

    # ── Mode 2: Synthetic stereo demo ──
    elif args.image:
        print("=" * 50)
        print("  Synthetic Stereo Demo")
        print("=" * 50)

        img = cv2.imread(args.image)
        if img is None:
            print(f"  Error: cannot read {args.image}")
            return
        h, w = img.shape[:2]
        print(f"  Image: {args.image} ({w}x{h})")

        # Generate synthetic stereo
        print("  Generating synthetic stereo pair...")
        left, right = generate_synthetic_stereo_pair(args.image, baseline_px=40)

        print("  Creating synthetic calibration...")
        stereo = generate_synthetic_calibration(image_size=(w, h))
        estimator = StereoDepthEstimator(stereo)

        # Rectify (identity for synthetic, but tests the pipeline)
        left_rect, right_rect = stereo.rectify_pair(left, right)

        # Compute disparity
        print("  Computing disparity...")
        t0 = time.time()
        disparity = estimator.compute_disparity(left_rect, right_rect)
        dt = time.time() - t0
        print(f"  Disparity computed in {dt*1000:.0f}ms")

        valid = disparity[disparity > 0]
        if len(valid) > 0:
            print(f"  Disparity range: {valid.min():.1f} — {valid.max():.1f} px")
            print(f"  Valid pixels: {len(valid)} / {disparity.size} ({100*len(valid)/disparity.size:.1f}%)")

        # Test 3D point at image centre
        cx, cy = w // 2, h // 2
        pt = estimator.get_3d_point(disparity, cx, cy)
        if pt:
            print(f"  Centre point 3D: X={pt['x_mm']}, Y={pt['y_mm']}, Z={pt['z_mm']} mm ({pt['confidence']})")
        else:
            print("  Centre point: no valid disparity")

        # Save outputs
        epipolar = draw_epipolar_lines(left_rect, right_rect)
        cv2.imwrite(os.path.join(args.output, "epipolar.jpg"), epipolar, [cv2.IMWRITE_JPEG_QUALITY, 95])

        depth_vis = draw_depth_overlay(left_rect, disparity)
        cv2.imwrite(os.path.join(args.output, "depth_overlay.jpg"), depth_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

        cv2.imwrite(os.path.join(args.output, "left.jpg"), left, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(args.output, "right.jpg"), right, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n  Outputs saved to {args.output}/")

    # ── Mode 3: Process stereo pair with calibration ──
    elif args.left and args.right:
        print("=" * 50)
        print("  Stereo 3D Processing")
        print("=" * 50)

        left_img = cv2.imread(args.left)
        right_img = cv2.imread(args.right)
        if left_img is None or right_img is None:
            print("  Error: cannot read stereo images")
            return

        h, w = left_img.shape[:2]
        print(f"  Images: {w}x{h}")

        if args.calibration:
            print(f"  Loading calibration: {args.calibration}")
            cam_l = CameraCalibrator()
            cam_r = CameraCalibrator()
            stereo = StereoCalibrator(cam_l, cam_r)
            stereo.load(args.calibration)
        else:
            print("  No calibration file — using synthetic parameters")
            stereo = generate_synthetic_calibration(image_size=(w, h))

        estimator = StereoDepthEstimator(stereo)

        # Rectify
        left_rect, right_rect = stereo.rectify_pair(left_img, right_img)

        # Disparity
        t0 = time.time()
        disparity = estimator.compute_disparity(left_rect, right_rect)
        dt = time.time() - t0
        print(f"  Disparity: {dt*1000:.0f}ms")

        # Save outputs
        epipolar = draw_epipolar_lines(left_rect, right_rect)
        cv2.imwrite(os.path.join(args.output, "epipolar.jpg"), epipolar, [cv2.IMWRITE_JPEG_QUALITY, 95])

        depth_vis = draw_depth_overlay(left_rect, disparity)
        cv2.imwrite(os.path.join(args.output, "depth_overlay.jpg"), depth_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Outputs saved to {args.output}/")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
