#!/usr/bin/env python3
"""
Contact Angle Analyzer - Automatic Sessile Drop Analysis System
================================================================
Upload a droplet image and automatically perform image enhancement,
droplet detection, baseline localization, profile fitting, and
contact angle calculation with visualization.

Usage:
    python contact_angle_analyzer.py <image_path>
    python contact_angle_analyzer.py <image_path> --output result.png
    python contact_angle_analyzer.py <image_path> --method circle
    python contact_angle_analyzer.py <image_path> --no-show
    python contact_angle_analyzer.py --generate test.png --angle 110

Pipeline:
    1. Image enhancement (CLAHE, denoising, sharpening)
    2. Droplet segmentation (Otsu + morphological ops)
    3. Baseline detection (convex hull + width scan + RANSAC)
    4. Contact angle calculation (circle fit, polynomial fit, ellipse fit)
    5. Result visualization (6-panel report)
"""

import cv2
import numpy as np
from scipy import optimize, ndimage
from scipy.optimize import curve_fit, minimize
import matplotlib
matplotlib.use('Agg')  # non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
import matplotlib.patches as mpatches
import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Fix encoding on Windows without crashing redirected stdout
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'buffer'):
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            import io
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass  # stdout is redirected (e.g. IDE), skip re-wrapping


# ============================================================================
# Module 1: Image Enhancer
# ============================================================================

class ImageEnhancer:
    """Image enhancement -- improve droplet image quality for segmentation."""

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load image, supports paths with non-ASCII characters."""
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img

    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    @staticmethod
    def denoise_bilateral(gray: np.ndarray, d: int = 7,
                          sigma_color: float = 50,
                          sigma_space: float = 50) -> np.ndarray:
        """Bilateral filter -- edge-preserving denoising."""
        return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

    @staticmethod
    def denoise_gaussian(gray: np.ndarray, kernel: int = 5) -> np.ndarray:
        """Gaussian filter -- mild denoising."""
        return cv2.GaussianBlur(gray, (kernel, kernel), 0)

    @staticmethod
    def clahe(gray: np.ndarray, clip_limit: float = 2.0,
              tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """CLAHE -- adaptive histogram equalization for uneven lighting."""
        clahe_obj = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe_obj.apply(gray)

    @staticmethod
    def gamma_correction(gray: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Gamma correction -- adjust overall brightness."""
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)]).astype(np.uint8)
        return cv2.LUT(gray, table)

    @staticmethod
    def sharpen(gray: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Unsharp mask sharpening."""
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)

    @staticmethod
    def auto_enhance(img: np.ndarray) -> np.ndarray:
        """Auto enhancement pipeline: gray -> denoise -> CLAHE -> sharpen."""
        gray = ImageEnhancer.to_gray(img)
        denoised = ImageEnhancer.denoise_bilateral(
            gray, d=5, sigma_color=30, sigma_space=30)
        enhanced = ImageEnhancer.clahe(
            denoised, clip_limit=2.5, tile_grid_size=(8, 8))
        return ImageEnhancer.sharpen(enhanced, strength=0.3)


# ============================================================================
# Module 2: Droplet Detector
# ============================================================================

class DropletDetector:
    """Droplet detection -- segment the droplet from the enhanced image."""

    @staticmethod
    def otsu_threshold(gray: np.ndarray) -> np.ndarray:
        """
        Otsu automatic thresholding.

        Tests both dark-droplet/light-background and light-droplet/dark-background
        modes. Selects the one whose area ratio is closer to a target of 25%
        (typical droplet occupies 2%-85% of the image).
        """
        h, w = gray.shape
        img_area = h * w

        _, mask_dark = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, mask_bright = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cnt_dark = np.count_nonzero(mask_dark)
        cnt_bright = np.count_nonzero(mask_bright)
        ratio_dark = cnt_dark / img_area
        ratio_bright = cnt_bright / img_area

        target_ratio = 0.25
        dark_ok = 0.02 < ratio_dark < 0.85
        bright_ok = 0.02 < ratio_bright < 0.85

        if dark_ok and not bright_ok:
            return mask_dark
        if bright_ok and not dark_ok:
            return mask_bright
        if dark_ok and bright_ok:
            if abs(ratio_dark - target_ratio) < abs(ratio_bright - target_ratio):
                return mask_dark
            return mask_bright
        # fallback: assume dark droplet (transmission lighting)
        return mask_dark

    @staticmethod
    def morphological_clean(mask: np.ndarray, open_kernel: int = 5,
                            close_kernel: int = 9) -> np.ndarray:
        """Morphological cleanup -- remove noise, fill holes."""
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    @staticmethod
    def fill_holes(mask: np.ndarray) -> np.ndarray:
        """Fill internal holes in the mask."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)
        return filled

    @staticmethod
    def edge_detect(gray: np.ndarray, low: int = 30,
                    high: int = 100) -> np.ndarray:
        """Canny edge detection -- fallback for thresholding."""
        edges = cv2.Canny(gray, low, high)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        cv2.drawContours(mask, contours, -1, 255, -1)
        return mask

    @staticmethod
    def find_droplet(mask: np.ndarray) -> tuple:
        """
        Find the main droplet contour (largest connected component).

        Returns: (contour, clean_mask)
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise RuntimeError("No contour detected -- check image quality")

        h, w = mask.shape
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if 0.005 * h * w < area < 0.9 * h * w:
                valid_contours.append(c)

        if not valid_contours:
            valid_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        droplet_contour = max(valid_contours, key=cv2.contourArea)

        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [droplet_contour], -1, 255, -1)
        return droplet_contour, clean_mask

    @staticmethod
    def detect(img: np.ndarray) -> tuple:
        """
        Full detection pipeline.

        Returns: (contour, mask, enhanced_gray)
        """
        enhanced = ImageEnhancer.auto_enhance(img)

        # Plan A: Otsu + morphology
        mask_otsu = DropletDetector.otsu_threshold(enhanced)
        mask_otsu = DropletDetector.morphological_clean(mask_otsu)
        mask_otsu = DropletDetector.fill_holes(mask_otsu)

        # Plan B: Canny edge detection (fallback)
        mask_edge = DropletDetector.edge_detect(enhanced)

        cnt_otsu = np.count_nonzero(mask_otsu)
        cnt_edge = np.count_nonzero(mask_edge)
        h, w = enhanced.shape
        img_area = h * w

        if 0.02 * img_area < cnt_otsu < 0.95 * img_area:
            mask = mask_otsu
        elif 0.02 * img_area < cnt_edge < 0.95 * img_area:
            mask = mask_edge
        else:
            mask = mask_otsu if cnt_otsu > cnt_edge else mask_edge

        contour, final_mask = DropletDetector.find_droplet(mask)
        return contour, final_mask, enhanced


# ============================================================================
# Module 3: Baseline Detector
# ============================================================================

class BaselineDetector:
    """
    Baseline detection -- find the solid-liquid contact line.

    Multi-level fallback strategy:
        1. Convex hull bottom edge (most robust, reflection-resistant)
        2. Width scan (widest point of droplet = contact line)
        3. Bottom points RANSAC line fitting
        4. Lowest contour point fallback
    """

    @staticmethod
    def detect_by_convex_hull(contour: np.ndarray,
                              image_shape: tuple) -> tuple:
        """
        Convex hull bottom edge method.
        The bottom horizontal edge of the droplet's convex hull is the baseline.

        Returns: (slope, intercept) or None
        """
        h, w = image_shape
        hull = cv2.convexHull(contour)
        hull_pts = hull.reshape(-1, 2)
        n = len(hull_pts)

        if n < 3:
            return None

        y_median = np.median(hull_pts[:, 1])
        best_edge_pts = None
        best_edge_length = 0

        for i in range(n):
            p1 = hull_pts[i]
            p2 = hull_pts[(i + 1) % n]
            length = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            avg_y = (p1[1] + p2[1]) / 2

            if abs(p2[0] - p1[0]) < 1e-6:
                slope_edge = 999.0
            else:
                slope_edge = abs((p2[1] - p1[1]) / (p2[0] - p1[0]))

            if (avg_y > y_median and slope_edge < 0.2 and
                    length > best_edge_length and length > w * 0.08):
                best_edge_length = length
                best_edge_pts = (p1, p2)

        if best_edge_pts is None:
            return None

        p1, p2 = best_edge_pts
        if abs(p2[0] - p1[0]) < 1e-6:
            slope = 0.0
            intercept = (p1[1] + p2[1]) / 2
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            intercept = p1[1] - slope * p1[0]

        return (slope, intercept)

    @staticmethod
    def get_bottom_points(contour: np.ndarray,
                          fraction: float = 0.10) -> np.ndarray:
        """Extract the lowest fraction of contour points."""
        points = contour.reshape(-1, 2)
        y_coords = points[:, 1]
        threshold_y = np.percentile(y_coords, 100 * (1 - fraction))
        return points[y_coords >= threshold_y]

    @staticmethod
    def ransac_line(points: np.ndarray, n_iterations: int = 200,
                    threshold: float = 3.0) -> tuple:
        """
        RANSAC line fitting.

        Returns: (slope, intercept) or None
        """
        if len(points) < 2:
            return None

        best_inliers = 0
        best_line = None

        for _ in range(n_iterations):
            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]

            if abs(p2[0] - p1[0]) < 1e-6:
                continue

            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            intercept = p1[1] - slope * p1[0]

            denom = np.sqrt(slope ** 2 + 1)
            distances = np.abs(
                points[:, 1] - (slope * points[:, 0] + intercept)) / denom
            n_inliers = np.sum(distances < threshold)

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_line = (slope, intercept)

        if best_line is not None:
            slope, intercept = best_line
            denom = np.sqrt(slope ** 2 + 1)
            distances = np.abs(
                points[:, 1] - (slope * points[:, 0] + intercept)) / denom
            inliers = points[distances < threshold * 1.5]

            if len(inliers) >= 2:
                A = np.column_stack([inliers[:, 0], np.ones(len(inliers))])
                b = inliers[:, 1]
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                best_line = (x[0], x[1])

        return best_line

    @staticmethod
    def detect(contour: np.ndarray, image_shape: tuple) -> tuple:
        """
        Detect the baseline with multi-level fallback.

        Returns: (slope, intercept)
        """
        h, w = image_shape

        # Method 1: Convex hull bottom edge (preferred)
        result = BaselineDetector.detect_by_convex_hull(contour, image_shape)
        if result is not None:
            slope, intercept = result
            if abs(slope) < 0.3 and 0.05 * h < intercept < 0.98 * h:
                return slope, intercept

        # Method 2: Width scan
        points = contour.reshape(-1, 2)
        y_min = int(points[:, 1].min())
        y_max = int(points[:, 1].max())
        y_range = y_max - y_min

        if y_range > 5:
            best_y = y_max
            best_width = 0
            scan_start = y_max - int(y_range * 0.05)
            scan_end = y_min + int(y_range * 0.15)

            for y_test in range(scan_start, scan_end, -1):
                nearby = points[np.abs(points[:, 1] - y_test) < 3]
                if len(nearby) >= 2:
                    width = nearby[:, 0].max() - nearby[:, 0].min()
                    if width > best_width:
                        best_width = width
                        best_y = y_test

            if best_width > 10 and 0.05 * h < best_y < 0.98 * h:
                refine_pts = points[np.abs(points[:, 1] - best_y) < 8]
                if len(refine_pts) >= 5:
                    return (0.0, float(np.mean(refine_pts[:, 1])))
                return (0.0, float(best_y))

        # Method 3: Bottom points RANSAC
        bottom_pts = BaselineDetector.get_bottom_points(contour, fraction=0.10)
        line = BaselineDetector.ransac_line(bottom_pts)
        if line is not None:
            slope, intercept = line
            if abs(slope) < 0.3:
                return slope, intercept
            return (0.0, float(np.mean(bottom_pts[:, 1])))

        # Method 4: Fallback
        y_bottom = np.max(points[:, 1])
        return (0.0, float(y_bottom))


# ============================================================================
# Module 4: Contact Angle Calculator
# ============================================================================

class ContactAngleCalculator:
    """
    Contact angle calculation module.

    Method 1 - Circle Fit (default, recommended):
        Least-squares circle fit to the droplet arc profile.
        Based on the spherical cap assumption (Young-Laplace).
        theta = arccos(d / R), where d = distance from circle center to
        baseline, R = circle radius.

    Method 2 - Polynomial Fit:
        Fit a 2nd/3rd order polynomial to points near the contact point.
        Calculate the tangent angle at the contact point.

    Method 3 - Ellipse Fit (experimental):
        Ellipse fit to the arc profile. For gravity-deformed droplets.
    """

    # ------------------------------------------------------------------
    # Contact point localization
    # ------------------------------------------------------------------

    @staticmethod
    def find_contact_points(contour: np.ndarray, slope: float,
                            intercept: float, tolerance: float = 5.0) -> list:
        """
        Find where the droplet contour intersects the baseline.

        Returns: [(x_left, y_left), (x_right, y_right)]
        """
        points = contour.reshape(-1, 2)

        if abs(slope) < 1e-6:
            distances = np.abs(points[:, 1] - intercept)
        else:
            denom = np.sqrt(slope ** 2 + 1)
            distances = np.abs(
                points[:, 1] - (slope * points[:, 0] + intercept)) / denom

        near = points[distances < tolerance]
        if len(near) < 2:
            near = points[distances < tolerance * 3]
        if len(near) < 2:
            near = points[distances < tolerance * 6]

        if len(near) >= 2:
            near = near[near[:, 0].argsort()]
            return [(float(near[0][0]), float(near[0][1])),
                    (float(near[-1][0]), float(near[-1][1]))]

        # Fallback: use bottom-most left and right points
        contour_center_x = np.median(points[:, 0])
        y_bottom = np.max(points[:, 1])
        bottom_pts = points[points[:, 1] >= y_bottom - 8]
        left_candidates = bottom_pts[bottom_pts[:, 0] < contour_center_x]
        right_candidates = bottom_pts[bottom_pts[:, 0] > contour_center_x]

        if len(left_candidates) > 0 and len(right_candidates) > 0:
            left_pt = left_candidates[left_candidates[:, 0].argmax()]
            right_pt = right_candidates[right_candidates[:, 0].argmin()]
        else:
            left_pt = bottom_pts[bottom_pts[:, 0].argmin()]
            right_pt = bottom_pts[bottom_pts[:, 0].argmax()]

        return [(float(left_pt[0]), float(left_pt[1])),
                (float(right_pt[0]), float(right_pt[1]))]

    # ------------------------------------------------------------------
    # Method 1: Circle Fit (primary)
    # ------------------------------------------------------------------

    @staticmethod
    def circle_fit_angle(contour: np.ndarray, slope: float,
                         intercept: float) -> tuple:
        """
        Circle fit method -- fit a circle to the droplet arc (liquid-vapor
        interface only, excluding the solid-liquid interface at the baseline).

        Returns: (angle_deg, (cx, cy, R)) or (None, None)
        """
        points = contour.reshape(-1, 2).astype(np.float64)

        if len(points) < 6:
            return None, None

        # Exclude points near the baseline (solid-liquid interface)
        if abs(slope) < 1e-6:
            dist_to_baseline = np.abs(points[:, 1] - intercept)
        else:
            denom = np.sqrt(slope ** 2 + 1)
            dist_to_baseline = np.abs(
                points[:, 1] - (slope * points[:, 0] + intercept)) / denom

        arc_mask = dist_to_baseline > 3.0
        arc_points = points[arc_mask]

        if len(arc_points) < 10:
            arc_mask = dist_to_baseline > 1.5
            arc_points = points[arc_mask]
        if len(arc_points) < 6:
            arc_points = points  # last resort

        x = arc_points[:, 0]
        y = arc_points[:, 1]

        # Least-squares circle fit: x^2 + y^2 + Ax + By + C = 0
        # cx = -A/2, cy = -B/2, R = sqrt(cx^2 + cy^2 - C)
        A_mat = np.column_stack([x, y, np.ones_like(x)])
        b_vec = -(x ** 2 + y ** 2)

        try:
            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            A, B, C = sol

            cx = -A / 2.0
            cy = -B / 2.0
            R_sq = cx ** 2 + cy ** 2 - C

            if R_sq <= 0:
                return None, None
            R = np.sqrt(R_sq)

            if R < 5:
                return None, None

            # Distance from circle center to baseline
            if abs(slope) < 1e-6:
                d = abs(cy - intercept)
            else:
                denom = np.sqrt(slope ** 2 + 1)
                d = abs(slope * cx - cy + intercept) / denom

            if d > R:
                d = min(d, R * 0.9999)

            cos_theta = np.clip(d / R, -1.0, 1.0)
            theta_base = np.degrees(np.arccos(cos_theta))

            # Determine if center is above or below baseline (image y-down coords)
            baseline_y_at_cx = slope * cx + intercept

            if cy < baseline_y_at_cx:
                # Center above baseline -> hydrophobic, theta > 90
                angle_deg = 180.0 - theta_base
            else:
                # Center below baseline -> hydrophilic, theta < 90
                angle_deg = theta_base

            return angle_deg, (cx, cy, R)

        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            return None, None

    # ------------------------------------------------------------------
    # Method 2: Polynomial Fit
    # ------------------------------------------------------------------

    @staticmethod
    def extract_profile_near_contact(contour: np.ndarray,
                                     contact_point: tuple,
                                     side: str,
                                     n_points: int = 20) -> np.ndarray:
        """
        Extract contour points near the contact point, along the droplet arc.

        side: 'left' or 'right'
        """
        points = contour.reshape(-1, 2)
        cx, cy = contact_point

        if side == 'left':
            candidates = points[
                (points[:, 0] >= cx - 2) &
                (points[:, 0] <= cx + 60) &
                (points[:, 1] <= cy + 3)
            ]
        else:
            candidates = points[
                (points[:, 0] <= cx + 2) &
                (points[:, 0] >= cx - 60) &
                (points[:, 1] <= cy + 3)
            ]

        if len(candidates) < 5:
            dists = np.sqrt(
                (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
            candidates = points[dists < 50]

        if len(candidates) < 5:
            return np.array([])

        dists = np.sqrt(
            (candidates[:, 0] - cx) ** 2 + (candidates[:, 1] - cy) ** 2)
        sorted_idx = np.argsort(dists)
        candidates = candidates[sorted_idx]

        return candidates[:n_points]

    @staticmethod
    def polynomial_angle(points: np.ndarray, contact_point: tuple,
                         order: int = 2) -> float:
        """
        Polynomial fit method.

        1. Transform points to math coords with contact point as origin (y up)
        2. Fit y = a1*x + a2*x^2 (+ a3*x^3)
        3. dy/dx at x=0 = a1
        4. Contact angle = arctan(|a1|)
        """
        if len(points) < order + 2:
            return None

        cx, cy = contact_point
        x_rel = points[:, 0] - cx
        y_rel = -(points[:, 1] - cy)  # image y -> math y (upward positive)

        try:
            coeffs = np.polyfit(x_rel, y_rel, order)
            # polyfit returns [a_n, ..., a_1, a_0]
            # derivative at x=0 = a_1 = coeffs[-2]
            derivative = coeffs[-2]
            angle_rad = np.arctan(np.abs(derivative))
            return np.degrees(angle_rad)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Method 3: Ellipse Fit (experimental)
    # ------------------------------------------------------------------

    @staticmethod
    def ellipse_fit_angle(contour: np.ndarray, slope: float,
                          intercept: float) -> float:
        """
        Ellipse fit method (experimental) -- fits the arc portion only.
        Note: Less accurate than circle fit. Useful only when the droplet
        is visibly non-spherical (gravity-deformed).
        """
        points = contour.reshape(-1, 2)
        if len(points) < 6:
            return None

        # Use arc points only (same logic as circle fit)
        if abs(slope) < 1e-6:
            dist_to_baseline = np.abs(points[:, 1] - intercept)
        else:
            denom = np.sqrt(slope ** 2 + 1)
            dist_to_baseline = np.abs(
                points[:, 1] - (slope * points[:, 0] + intercept)) / denom

        arc_points = points[dist_to_baseline > 3.0]
        if len(arc_points) < 6:
            arc_points = points[dist_to_baseline > 1.5]
        if len(arc_points) < 6:
            return None

        try:
            ellipse = cv2.fitEllipse(arc_points.astype(np.float32))
            (cx, cy), (major, minor), _ = ellipse
            a = major / 2.0
            b = minor / 2.0

            if abs(slope) < 1e-6:
                d = abs(cy - intercept)
            else:
                denom = np.sqrt(slope ** 2 + 1)
                d = abs(slope * cx - cy + intercept) / denom

            b_eff = min(a, b)
            if d > b_eff:
                return None

            cos_theta = np.clip(d / b_eff, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_theta))

            baseline_y_at_cx = slope * cx + intercept
            if cy < baseline_y_at_cx:
                angle_deg = 180.0 - angle_deg

            return np.clip(angle_deg, 0.0, 180.0)

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main calculation entry point
    # ------------------------------------------------------------------

    @staticmethod
    def calculate(contour: np.ndarray, slope: float, intercept: float,
                  method: str = 'circle') -> dict:
        """
        Main calculation interface.

        Args:
            method: 'circle' | 'polynomial' | 'ellipse' | 'all'

        Returns:
            dict with keys: left_angle, right_angle, circle_angle,
            ellipse_angle, mean_angle, contact_points, circle_params, method
        """
        result = {
            'left_angle': None,
            'right_angle': None,
            'circle_angle': None,
            'ellipse_angle': None,
            'mean_angle': None,
            'contact_points': [],
            'circle_params': None,
            'method': method,
        }

        # Find contact points
        contact_pts = ContactAngleCalculator.find_contact_points(
            contour, slope, intercept)
        result['contact_points'] = contact_pts
        left_pt, right_pt = contact_pts[0], contact_pts[1]

        # Circle fit (always computed as reference)
        circle_angle, circle_params = ContactAngleCalculator.circle_fit_angle(
            contour, slope, intercept)
        result['circle_angle'] = circle_angle
        result['circle_params'] = circle_params

        # Polynomial fit
        if method in ('polynomial', 'all'):
            for side, pt in [('left', left_pt), ('right', right_pt)]:
                profile = ContactAngleCalculator.extract_profile_near_contact(
                    contour, pt, side, n_points=30)
                angle = None
                if len(profile) >= 4:
                    angle = ContactAngleCalculator.polynomial_angle(
                        profile, pt, order=2)
                    if angle is None:
                        angle = ContactAngleCalculator.polynomial_angle(
                            profile, pt, order=3)
                if side == 'left':
                    result['left_angle'] = angle
                else:
                    result['right_angle'] = angle

        # Ellipse fit
        if method in ('ellipse', 'all'):
            result['ellipse_angle'] = \
                ContactAngleCalculator.ellipse_fit_angle(
                    contour, slope, intercept)

        # Aggregate mean angle (prefer circle fit)
        angles_poly = []
        if result['left_angle'] is not None:
            angles_poly.append(result['left_angle'])
        if result['right_angle'] is not None:
            angles_poly.append(result['right_angle'])

        if method in ('circle', 'all'):
            if result['circle_angle'] is not None:
                result['mean_angle'] = result['circle_angle']
            elif angles_poly:
                result['mean_angle'] = np.mean(angles_poly)
            elif result['ellipse_angle'] is not None:
                result['mean_angle'] = result['ellipse_angle']
        elif method == 'polynomial':
            if angles_poly:
                result['mean_angle'] = np.mean(angles_poly)
            elif result['circle_angle'] is not None:
                result['mean_angle'] = result['circle_angle']
        else:
            result['mean_angle'] = result.get('ellipse_angle')

        return result


# ============================================================================
# Module 5: Visualizer
# ============================================================================

class Visualizer:
    """Visualization -- generate annotated result images."""

    @staticmethod
    def draw(img: np.ndarray, contour: np.ndarray, mask: np.ndarray,
             result: dict, slope: float, intercept: float,
             output_path: str = None):
        """Draw the complete 6-panel visualization figure."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Contact Angle Analysis Report',
                     fontsize=16, fontweight='bold')

        # (0,0): Original image
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')

        # (0,1): Enhanced image
        enhanced = ImageEnhancer.auto_enhance(img)
        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title('Enhanced (CLAHE + Denoise + Sharpen)', fontsize=12)
        axes[0, 1].axis('off')

        # (0,2): Segmentation mask
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title('Droplet Segmentation (Otsu + Morphology)',
                             fontsize=12)
        axes[0, 2].axis('off')

        # (1,0): Detection overlay
        h, w = img.shape[:2]
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

        # Draw contour
        cv2.drawContours(display_img, [contour], -1, (0, 255, 255), 2)

        # Draw baseline
        if abs(slope) < 1e-6:
            y_base = int(intercept)
            cv2.line(display_img, (0, y_base), (w, y_base), (255, 0, 0), 2)
        else:
            x1, x2 = 0, w
            y1 = int(slope * x1 + intercept)
            y2 = int(slope * x2 + intercept)
            cv2.line(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw contact points
        for pt in result['contact_points']:
            cv2.circle(display_img, (int(pt[0]), int(pt[1])),
                       8, (255, 0, 0), -1)
            cv2.circle(display_img, (int(pt[0]), int(pt[1])),
                       10, (255, 255, 255), 1)

        # Draw tangent lines at contact points
        for idx, pt in enumerate(result['contact_points']):
            side = 'left' if idx == 0 else 'right'
            angle = result['left_angle'] if idx == 0 else result['right_angle']
            if angle is not None:
                cx_px, cy_px = int(pt[0]), int(pt[1])
                tangent_len = 60
                base_angle_rad = np.arctan(slope)
                angle_rad = np.radians(angle)

                if side == 'left':
                    tangent_angle = base_angle_rad - angle_rad
                else:
                    tangent_angle = base_angle_rad + angle_rad

                dx = int(tangent_len * np.cos(tangent_angle))
                dy = int(tangent_len * np.sin(tangent_angle))
                cv2.line(display_img, (cx_px, cy_px),
                         (cx_px + dx, cy_px + dy), (0, 255, 0), 2)

                label_x = cx_px + int(30 * np.cos(tangent_angle + 0.3))
                label_y = cy_px + int(30 * np.sin(tangent_angle + 0.3))
                cv2.putText(display_img, f'{angle:.1f}deg',
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw fitted circle
        if result.get('circle_params') is not None:
            cx_c, cy_c, R = result['circle_params']
            cx_c, cy_c = int(cx_c), int(cy_c)
            R = int(R)
            if R < max(h, w):
                cv2.circle(display_img, (cx_c, cy_c), R, (0, 0, 255), 1)
                cv2.circle(display_img, (cx_c, cy_c), 4, (0, 0, 255), -1)

        axes[1, 0].imshow(display_img)
        axes[1, 0].set_title(
            'Detection: Contour + Baseline + Contact Points + Tangent',
            fontsize=12)
        axes[1, 0].axis('off')

        # (1,1): Zoomed view
        x_bb, y_bb, bw, bh = cv2.boundingRect(contour)
        margin = 20
        x_crop = max(0, x_bb - margin)
        y_crop = max(0, y_bb - margin)
        w_crop = min(w - x_crop, bw + 2 * margin)
        h_crop = min(h - y_crop, bh + 2 * margin)

        zoom_img = display_img[
            y_crop:y_crop + h_crop, x_crop:x_crop + w_crop].copy()
        axes[1, 1].imshow(zoom_img)
        axes[1, 1].set_title('Zoom: Contact Angle Measurement', fontsize=12)
        axes[1, 1].axis('off')

        # (1,2): Measurement report panel
        axes[1, 2].axis('off')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)

        lines = []
        lines.append('=' * 34)
        lines.append('    Contact Angle Measurement Report')
        lines.append('=' * 34)
        lines.append('')

        if result['left_angle'] is not None:
            lines.append(
                f'  Left  (polynomial):   {result["left_angle"]:.2f} deg')
        else:
            lines.append('  Left  (polynomial):   N/A')

        if result['right_angle'] is not None:
            lines.append(
                f'  Right (polynomial):   {result["right_angle"]:.2f} deg')
        else:
            lines.append('  Right (polynomial):   N/A')

        lines.append('')

        if result['circle_angle'] is not None:
            lines.append(
                f'  Circle fit:           {result["circle_angle"]:.2f} deg')

        if result['ellipse_angle'] is not None:
            lines.append(
                f'  Ellipse fit:          {result["ellipse_angle"]:.2f} deg')

        lines.append('-' * 34)

        if result['mean_angle'] is not None:
            mean_angle = result['mean_angle']
            lines.append('')
            lines.append(f'  *** Mean Contact Angle: {mean_angle:.2f} deg')
            lines.append('')

            if mean_angle < 10:
                wetting = 'Superhydrophilic'
            elif mean_angle < 90:
                wetting = 'Hydrophilic'
            elif mean_angle < 150:
                wetting = 'Hydrophobic'
            else:
                wetting = 'Superhydrophobic'

            lines.append(f'  Surface:              {wetting}')

        lines.append('')
        lines.append('=' * 34)

        text = '\n'.join(lines)
        axes[1, 2].text(0.05, 0.98, text, transform=axes[1, 2].transAxes,
                        fontsize=10, fontfamily='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat',
                                  alpha=0.5))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n[OK] Result saved to: {output_path}")

        return fig


# ============================================================================
# Module 6: Main Analyzer
# ============================================================================

class ContactAngleAnalyzer:
    """Main controller -- integrates all modules into a unified interface."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_img = None
        self.enhanced_gray = None
        self.contour = None
        self.mask = None
        self.slope = None
        self.intercept = None
        self.result = None

    def analyze(self, method: str = 'circle') -> dict:
        """
        Run the complete analysis pipeline.

        Args:
            method: 'circle' (default, recommended), 'polynomial', or 'all'

        Returns:
            dict with all measurement results
        """
        print("=" * 60)
        print("  Contact Angle Analyzer v1.0")
        print("=" * 60)

        # Step 1: Load image
        print(f"\n[1/5] Loading image: {self.image_path}")
        self.original_img = ImageEnhancer.load_image(self.image_path)
        h, w = self.original_img.shape[:2]
        print(f"       Size: {w} x {h}")

        # Step 2: Image enhancement + droplet detection
        print("\n[2/5] Image enhancement and droplet segmentation...")
        self.contour, self.mask, self.enhanced_gray = \
            DropletDetector.detect(self.original_img)
        area = cv2.contourArea(self.contour)
        print(f"       Droplet area: {area:.0f} px2 "
              f"({100 * area / (h * w):.1f}% of image)")

        # Step 3: Baseline detection
        print("\n[3/5] Baseline detection...")
        self.slope, self.intercept = BaselineDetector.detect(
            self.contour, self.original_img.shape[:2])
        baseline_angle_deg = np.degrees(np.arctan(self.slope))
        print(f"       Baseline: y = {self.slope:.4f}x + {self.intercept:.1f}")
        print(f"       Baseline tilt: {baseline_angle_deg:.2f} deg")

        # Step 4: Contact angle calculation
        print(f"\n[4/5] Contact angle calculation (method: {method})...")
        self.result = ContactAngleCalculator.calculate(
            self.contour, self.slope, self.intercept, method)

        # Step 5: Output results
        print(f"\n[5/5] Analysis complete!")
        print(f"\n{'-' * 40}")
        print("  Measurement Results")
        print(f"{'-' * 40}")

        if self.result['left_angle'] is not None:
            print(f"  Left  (polynomial):  "
                  f"{self.result['left_angle']:.2f} deg")
        if self.result['right_angle'] is not None:
            print(f"  Right (polynomial):  "
                  f"{self.result['right_angle']:.2f} deg")
        if self.result['circle_angle'] is not None:
            print(f"  Circle fit:          "
                  f"{self.result['circle_angle']:.2f} deg")
        if self.result['ellipse_angle'] is not None:
            print(f"  Ellipse fit:         "
                  f"{self.result['ellipse_angle']:.2f} deg")

        if self.result['mean_angle'] is not None:
            mean = self.result['mean_angle']
            print(f"\n  *** Mean Contact Angle: {mean:.2f} deg")

            if mean < 10:
                wetting = 'Superhydrophilic'
            elif mean < 90:
                wetting = 'Hydrophilic'
            elif mean < 150:
                wetting = 'Hydrophobic'
            else:
                wetting = 'Superhydrophobic'

            print(f"  Surface property: {wetting}")

        print(f"{'-' * 40}\n")

        return self.result

    def visualize(self, output_path: str = None, show: bool = True):
        """Generate visualization."""
        if self.result is None:
            raise RuntimeError("Please call analyze() first")

        fig = Visualizer.draw(
            self.original_img, self.contour, self.mask,
            self.result, self.slope, self.intercept,
            output_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def save_report(self, output_dir: str = '.'):
        """Save analysis report (text + image)."""
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]

        img_path = os.path.join(output_dir, f'{base_name}_analysis.png')
        self.visualize(img_path, show=False)

        report_path = os.path.join(output_dir, f'{base_name}_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Contact Angle Measurement Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Image: {self.image_path}\n")
            f.write(
                f"Method: {self.result.get('method', 'circle')}\n\n")

            f.write("Results:\n")
            f.write("-" * 30 + "\n")
            if self.result['left_angle'] is not None:
                f.write(
                    f"Left  (polynomial):  {self.result['left_angle']:.2f} deg\n")
            if self.result['right_angle'] is not None:
                f.write(
                    f"Right (polynomial):  {self.result['right_angle']:.2f} deg\n")
            if self.result['circle_angle'] is not None:
                f.write(
                    f"Circle fit:          {self.result['circle_angle']:.2f} deg\n")
            if self.result['ellipse_angle'] is not None:
                f.write(
                    f"Ellipse fit:         {self.result['ellipse_angle']:.2f} deg\n")
            if self.result['mean_angle'] is not None:
                f.write(
                    f"\nMean Contact Angle: {self.result['mean_angle']:.2f} deg\n")

        print(f"[OK] Report saved to: {report_path}")

        return img_path, report_path


# ============================================================================
# Module 7: Sample Generator
# ============================================================================

class SampleGenerator:
    """
    Generate synthetic droplet images for algorithm testing and validation.

    Uses the spherical cap model (sessile drop, Young-Laplace).
    The generated image has a known contact angle for accuracy verification.
    """

    @staticmethod
    def generate(contact_angle: float = 75.0,
                 image_size: tuple = (600, 800),
                 noise_level: float = 0.02,
                 blur_sigma: float = 1.5,
                 add_reflection: bool = True) -> np.ndarray:
        """
        Generate a synthetic droplet image.

        Args:
            contact_angle: Target contact angle in degrees
            image_size: (height, width)
            noise_level: Gaussian noise std ratio
            blur_sigma: Edge blur amount
            add_reflection: Whether to add surface reflection

        Returns:
            BGR image as numpy array
        """
        h, w = image_size
        y_vals = np.linspace(0, 1, h)

        # Light gray gradient background
        background = np.ones((h, w)) * 0.85 + 0.1 * y_vals[:, np.newaxis]

        # Droplet parameters
        theta_rad = np.radians(contact_angle)
        drop_center_x = w // 2
        baseline_y = int(h * 0.72)
        base_radius = int(w * 0.22)

        # Spherical cap model
        if contact_angle <= 90:
            R = base_radius / np.sin(theta_rad)
            cy = baseline_y + R * np.cos(theta_rad)
        else:
            R = base_radius / np.sin(np.pi - theta_rad)
            cy = baseline_y - R * np.cos(np.pi - theta_rad)

        # Create droplet mask (above baseline only)
        droplet = np.zeros((h, w), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((j - drop_center_x) ** 2 + (i - cy) ** 2)
                if dist <= R and i <= baseline_y:
                    droplet[i, j] = 1.0

        # Add faint surface reflection (below baseline, won't affect threshold)
        if add_reflection:
            reflect_y_start = baseline_y
            reflect_y_end = baseline_y + int(h * 0.03)
            for i in range(reflect_y_start, min(reflect_y_end, h)):
                frac = (i - reflect_y_start) / max(
                    1, reflect_y_end - reflect_y_start)
                reflect_strength = 0.12 * (1 - frac)
                reflect_mask = droplet[baseline_y - 1, :] > 0.5
                droplet[i, reflect_mask] = reflect_strength

        # Droplet interior: dark gray (clear contrast from background)
        droplet_intensity = np.where(droplet > 0.5, 0.30, 0.0)
        droplet_intensity = np.where(
            (droplet > 0.01) & (droplet <= 0.5),
            0.78,  # faint reflection (near background, won't be mis-segmented)
            droplet_intensity)

        # Composite image
        img = background.copy()
        img = np.where(droplet > 0.01, droplet_intensity, img)

        # Apply Gaussian blur (simulates optical blur)
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur_sigma)

        # Add Gaussian noise
        noise = np.random.randn(h, w) * noise_level
        img = np.clip(img + noise, 0, 1)

        # Convert to 8-bit BGR
        img_8u = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)

        return img_bgr


# ============================================================================
# Command-line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Contact Angle Analyzer - Automatic Sessile Drop Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python contact_angle_analyzer.py droplet.jpg
  python contact_angle_analyzer.py droplet.jpg --output result.png
  python contact_angle_analyzer.py droplet.jpg --method circle
  python contact_angle_analyzer.py droplet.jpg --report
  python contact_angle_analyzer.py --generate sample.png --angle 110
        """)

    parser.add_argument('image', nargs='?',
                        help='Input image path')

    parser.add_argument('--output', '-o',
                        help='Output visualization image path')

    parser.add_argument('--method', '-m',
                        choices=['circle', 'polynomial', 'ellipse', 'all'],
                        default='circle',
                        help='Calculation method (default: circle, most accurate)')

    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the image window')

    parser.add_argument('--report', '-r', action='store_true',
                        help='Generate text report')

    parser.add_argument('--generate', '-g',
                        help='Generate a synthetic test image with known angle')

    parser.add_argument('--angle', '-a', type=float, default=75.0,
                        help='Target contact angle for synthetic image '
                             '(default: 75 deg)')

    args = parser.parse_args()

    # Generate mode
    if args.generate:
        print(f"Generating synthetic droplet image "
              f"(contact angle: {args.angle} deg)...")
        img = SampleGenerator.generate(
            contact_angle=args.angle,
            image_size=(600, 800),
            noise_level=0.015,
            add_reflection=True)
        cv2.imwrite(args.generate, img)
        print(f"[OK] Image saved to: {args.generate}")

        if args.image is None:
            args.image = args.generate
        else:
            return

    if args.image is None:
        parser.print_help()
        print("\nTip: Provide an image path, or use --generate to create a "
              "test image")
        sys.exit(1)

    # Analyze mode
    analyzer = ContactAngleAnalyzer(args.image)
    analyzer.analyze(method=args.method)

    # Visualization
    output_path = args.output
    if output_path is None and not args.no_show:
        output_path = os.path.splitext(args.image)[0] + '_result.png'

    analyzer.visualize(output_path=output_path, show=not args.no_show)

    # Generate report
    if args.report:
        analyzer.save_report(os.path.dirname(args.image) or '.')


if __name__ == '__main__':
    main()
