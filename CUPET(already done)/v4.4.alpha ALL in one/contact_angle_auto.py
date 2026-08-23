#!/usr/bin/env python3
"""
Contact Angle Auto-Measurement
===============================
Automatic sessile-drop contact angle measurement from a single image.

Pipeline (adapted from the reference implementation in AngleAP/):
  1. Multi-scale Hough circle detection to locate droplet candidates
  2. Select the best candidate (highest contrast, appropriate size)
  3. Crop tight ROI around droplet -> CLAHE enhance
  4. Re-detect droplet in ROI -> Otsu segment -> extract contour
  5. Adaptive baseline (contact line) detection
  6. Young-Laplace ODE fitting for contact angle
  7. Circle geometry + polynomial backup verification

Usage:
  python contact_angle_auto.py <image>
  python contact_angle_auto.py drop_image.jpg -o result.jpg
"""

import argparse, os, sys, warnings
import cv2, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

try:
    if hasattr(sys.stdout, 'buffer'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass


# ============================================================
#  Image enhancement
# ============================================================

def clahe_enhance(image, clip_limit=2.0, grid_size=8):
    """CLAHE in LAB + Gaussian sharpen."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_c, a_c, b_c = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)).apply(l_c)
    enhanced = cv2.cvtColor(cv2.merge((cl, a_c, b_c)), cv2.COLOR_LAB2BGR)
    gauss = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.2, gauss, -0.2, 0)


# ============================================================
#  Step 1: Find droplet candidates
# ============================================================

def find_candidates(gray, debug=True):
    """
    Multi-scale Hough circle detection.

    Uses multiple parameter sets to cover different droplet sizes,
    then deduplicates and returns unique candidates.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)

    # Parameter sets: (dp, minDist, param1, param2, minR, maxR)
    param_sets = [
        (1.0, 30, 50, 28, 15, 80),
        (1.0, 40, 45, 25, 20, 120),
        (1.2, 50, 55, 22, 25, 160),
        (1.2, 50, 60, 20, 30, 200),
    ]

    candidates = []
    seen = set()
    for dp, md, p1, p2, min_r, max_r in param_sets:
        circles = cv2.HoughCircles(
            enhanced, cv2.HOUGH_GRADIENT, dp=dp, minDist=md,
            param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            continue
        for c in np.round(circles[0, :]).astype("int"):
            cx_i, cy_i, r_i = int(c[0]), int(c[1]), int(c[2])
            # Skip edge-touching circles
            if cx_i - r_i < 5 or cx_i + r_i > w - 5:
                continue
            if cy_i - r_i < 5 or cy_i + r_i > h - 5:
                continue
            # Deduplicate
            key = (cx_i // 8, cy_i // 8, r_i // 8)
            if key not in seen:
                seen.add(key)
                candidates.append((cx_i, cy_i, r_i))

    if debug:
        print(f"  Found {len(candidates)} unique droplet candidates")
    return candidates


def score_candidate(gray, cx, cy, r):
    """
    Score a droplet candidate. Higher = more droplet-like.

    Criteria:
      - Strong edges at the circle boundary
      - Good contrast between inside and outside
      - Uniform interior (liquid is homogeneous)
      - Appropriate size (not too small, not too large)
    """
    h, w = gray.shape

    # Edge strength at circle boundary
    gy, gx = np.gradient(gray.astype(np.float64))
    grad = np.sqrt(gx ** 2 + gy ** 2)
    bdry = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(bdry, (cx, cy), r, 255, 2)
    edge_pts = grad[bdry > 0]
    edge_score = np.mean(edge_pts) if len(edge_pts) > 10 else 0

    # Interior
    in_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(in_mask, (cx, cy), max(3, r - 4), 255, -1)
    inside = gray[in_mask > 0]
    if len(inside) < 30:
        return -1

    # Outside ring
    out_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(out_mask, (cx, cy), min(w - 1, r + 10), 255, -1)
    cv2.circle(out_mask, (cx, cy), r + 3, 0, -1)
    outside = gray[out_mask > 0]
    if len(outside) < 20:
        return -1

    contrast = abs(np.mean(inside) - np.mean(outside))
    uniformity = max(0, 128 - np.std(inside))  # lower std = more uniform

    # Size: prefer r in [25, 150], peak at ~50
    if r < 20:
        size_score = 0
    elif r > 180:
        size_score = max(0, 1 - (r - 180) / 200)
    else:
        size_score = np.exp(-((r - 50) ** 2) / (2 * 60 ** 2))

    score = edge_score * 1.0 + contrast * 0.3 + uniformity * 0.2 + size_score * 50
    return score


def select_droplet(gray, candidates, debug=True):
    """Select the best droplet candidate by scoring."""
    if not candidates:
        raise RuntimeError(
            "No droplet candidates found.\n"
            "  Ensure the image contains a visible droplet with clear boundaries."
        )

    scored = [(score_candidate(gray, cx, cy, r), cx, cy, r)
              for cx, cy, r in candidates]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        raise RuntimeError("All candidates scored poorly.")

    if debug:
        print(f"  {len(scored)} candidates passed scoring")
        for i, (s, cx, cy, r) in enumerate(scored[:5]):
            in_mean = np.mean(gray[cy - r // 4:cy + r // 4,
                                   cx - r // 4:cx + r // 4])
            print(f"    #{i+1}: ({cx:4d},{cy:4d}) r={r:3d}  "
                  f"score={s:7.1f}  interior={in_mean:3.0f}")

    best = scored[0]
    cx, cy, r = best[1], best[2], best[3]
    if debug:
        print(f"  Selected: ({cx}, {cy}) r={r}")
    return cx, cy, r


# ============================================================
#  Step 2: Segment droplet within ROI
# ============================================================

def segment_in_roi(enhanced_roi, r_approx, debug=True):
    """
    Segment the droplet within a tight ROI.

    Steps:
      - Re-detect droplet position within ROI (refined center)
      - Create temporary baseline below the detected circle
      - Black out below temp baseline (removes substrate)
      - Otsu threshold to isolate droplet
      - Morphological cleanup
      - Extract largest contour
      - Detect real baseline (contact line)
    """
    h_roi, w_roi = enhanced_roi.shape[:2]
    gray = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2GRAY)

    # ---- Re-detect within ROI ----
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
        param1=30, param2=15,
        minRadius=max(3, r_approx // 3),
        maxRadius=r_approx * 2,
    )
    if circles is not None:
        cx_r, cy_r, r_r = np.round(circles[0, 0]).astype("int")
    else:
        cx_r, cy_r, r_r = w_roi // 2, h_roi // 2, r_approx

    # ---- Temporary baseline ----
    temp_bl = int(cy_r + r_r * 1.2)
    temp_bl = min(temp_bl, h_roi - 5)

    # ---- Black out below temp baseline ----
    gray_clean = gray.copy()
    gray_clean[temp_bl:, :] = 0

    # ---- Otsu ----
    _, thresh = cv2.threshold(gray_clean, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If >80% of the ROI is bright after thresholding, the droplet
    # is probably the dark region -> invert
    bright_frac = np.count_nonzero(thresh) / thresh.size
    if bright_frac > 0.8:
        thresh = cv2.bitwise_not(thresh)

    # ---- Morphology ----
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # ---- Extract contour ----
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in ROI segmentation.")

    cnt = max(contours, key=cv2.contourArea)
    contour = cnt.reshape(-1, 2)[:, [1, 0]]  # (x,y) -> (y,x)

    # ---- Real baseline ----
    baseline_y = _detect_baseline(contour, w_roi, temp_bl, debug)

    # ---- Points above baseline ----
    valid_mask = contour[:, 0] < baseline_y
    valid_contour = contour[valid_mask]

    if debug:
        print(f"  Contour: {len(contour)} total, {len(valid_contour)} above baseline")
        print(f"  Baseline y = {baseline_y}")

    return contour, baseline_y, valid_contour


def _detect_baseline(contour, img_w, fallback, debug):
    """Detect contact line from contour width profile."""
    unique_ys = np.unique(contour[:, 0])
    unique_ys = np.sort(unique_ys)[::-1]

    y_widths = []
    for y in unique_ys:
        xs = contour[contour[:, 0] == y, 1]
        if len(xs) > 1:
            y_widths.append((y, np.max(xs) - np.min(xs)))

    if not y_widths:
        return fallback

    max_w = max(w for _, w in y_widths)

    # Width collapse
    prev_w = 0
    for y, w_val in y_widths:
        if w_val < max_w * 0.8 and prev_w >= max_w * 0.8:
            if debug:
                print(f"  Baseline (collapse): y={y + 2}")
            return y + 2
        prev_w = w_val

    # Relaxed
    for y, w_val in y_widths:
        if w_val < max_w * 0.7:
            if debug:
                print(f"  Baseline (relaxed): y={y}")
            return y

    if debug:
        print(f"  Baseline (fallback): y={fallback}")
    return fallback


# ============================================================
#  Step 3: Young-Laplace fitting
# ============================================================

def calc_yl(valid_contour, baseline_y, r_guess, debug=True):
    """Young-Laplace equation fitting for contact angle."""
    if len(valid_contour) < 10:
        return _empty_yl_result()

    cy_p = valid_contour[:, 0].astype(np.float64)
    cx_p = valid_contour[:, 1].astype(np.float64)
    cz = baseline_y - cy_p

    idx_tip = np.argmax(cz)
    x_tip, z_tip = cx_p[idx_tip], cz[idx_tip]

    # Sort, clean gaps
    si = np.argsort(cx_p)
    sx, sz = cx_p[si], cz[si]
    dx = np.diff(sx)
    gaps = np.where(dx > 10)[0]
    if len(gaps) > 0:
        segs = []
        s0 = 0
        for g in gaps:
            segs.append((s0, g)); s0 = g + 1
        segs.append((s0, len(sx) - 1))
        longest = max(segs, key=lambda s: s[1] - s[0])
        sx, sz = sx[longest[0]:longest[1] + 1], sz[longest[0]:longest[1] + 1]

    # Trim bottom 15%
    zt = z_tip * 0.15
    vm = sz > zt
    xc, zc = sx[vm], sz[vm]

    if len(xc) < 10:
        return _empty_yl_result()

    idx_t = np.argmax(zc)
    xtc = xc[idx_t]
    xn = xc - xtc
    zn = zc - z_tip

    b0 = 2.0 / max(5.0, float(r_guess))
    al = _yl_side(xn, zn, 'left', b0)
    ar = _yl_side(xn, zn, 'right', b0)

    # Average: exclude sides that returned the default 90.0
    valid_sides = []
    if al != 90.0:
        valid_sides.append(al)
    if ar != 90.0:
        valid_sides.append(ar)
    aa = np.mean(valid_sides) if valid_sides else 90.0

    if debug:
        print(f"  Y-L: L={al:.2f}  R={ar:.2f}  Avg={aa:.2f} deg"
              + (" (single side)" if len(valid_sides) == 1 else ""))

    return {
        'angle_avg': round(float(aa), 2),
        'angle_left': round(float(al), 2),
        'angle_right': round(float(ar), 2),
        'tip': (float(x_tip), float(z_tip)),
        'valid': True, 'method': 'Young-Laplace',
        'x_norm': xn, 'z_norm': zn, 'x_tip_clean': xtc,
        'contour_x': cx_p, 'contour_z': cz,
    }


def _empty_yl_result():
    return {
        'angle_avg': 0.0, 'angle_left': 0.0, 'angle_right': 0.0,
        'tip': (0.0, 0.0), 'valid': False, 'method': 'Y-L (skipped)',
        'x_norm': None, 'z_norm': None, 'x_tip_clean': 0,
        'contour_x': None, 'contour_z': None,
    }


def _yl_side(xn, zn, side, b0):
    """Fit Y-L to one side."""
    if side == 'right':
        m = xn > 0; xs, zs = xn[m], zn[m]
    else:
        m = xn < 0; xs, zs = -xn[m], zn[m]

    if len(xs) < 8:
        return 90.0

    si = np.argsort(xs)
    xs, zs = xs[si], zs[si]
    zp = -zs

    def ode(s, y, b):
        x, z, phi = y
        if x < 1e-6:
            return [1.0, 0.0, b]
        return [np.cos(phi), np.sin(phi), b - np.sin(phi) / x]

    def obj(params):
        b = params[0]
        if b <= 0:
            return np.full(len(xs), 1e6)
        try:
            sol = solve_ivp(ode, (0, np.max(xs) * 1.5),
                            [1e-6, 0.0, 0.01], args=(b,),
                            method='RK45', dense_output=True)
            if not sol.success or len(sol.t) < 10:
                return np.full(len(xs), 1e6)
            return sol.sol(xs)[1] - zp
        except Exception:
            return np.full(len(xs), 1e6)

    try:
        res = least_squares(obj, [max(1e-9, b0)],
                            bounds=([1e-9], [np.inf]),
                            method='trf', ftol=1e-8, max_nfev=500)
        b_opt = res.x[0]
    except Exception:
        return 90.0

    try:
        sol = solve_ivp(ode, (0, np.max(xs)),
                        [1e-6, 0.0, 0.01], args=(b_opt,), method='RK45')
        phi = sol.y[2, -1]
        ang = np.degrees(phi)
        if ang < 0:
            ang += 180
        if ang > 180:
            ang = 360 - ang
        return float(ang)
    except Exception:
        return 90.0


# ============================================================
#  Step 4: Circle + Polynomial backup
# ============================================================

def calc_circle(valid_contour, baseline_y, debug=True):
    """Circle fit for geometric contact angle."""
    if len(valid_contour) < 10:
        return {'angle_avg': 90.0, 'angle_left': 90.0, 'angle_right': 90.0,
                'cx': 0, 'cy': 0, 'R': 0, 'valid': False}

    yp = valid_contour[:, 0].astype(np.float64)
    xp = valid_contour[:, 1].astype(np.float64)

    # Linearized circle: x^2 + y^2 = 2*cx*x + 2*cy*y + C
    z = xp ** 2 + yp ** 2
    A = np.column_stack([xp, yp, np.ones_like(xp)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        cx = coeffs[0] / 2.0
        cy = coeffs[1] / 2.0
        R = np.sqrt(max(0.0, coeffs[2] + cx ** 2 + cy ** 2))
        span = max(xp.max() - xp.min(), yp.max() - yp.min())
        if R <= 0 or R > span * 10:
            raise ValueError("Degenerate")

        # Contact angle: theta = arccos((baseline_y - cy) / R)
        # If |baseline_y - cy| > R, the circle doesn't reach the baseline
        # (flat droplet).  In that case theta -> 0 (perfect wetting).
        d_raw = (baseline_y - cy) / R
        if d_raw >= 1.0:
            theta = 0.0    # perfect wetting (superhydrophilic)
        elif d_raw <= -1.0:
            theta = 180.0  # perfect non-wetting
        else:
            theta = float(np.degrees(np.arccos(d_raw)))

        if debug:
            print(f"  Circle: center=({cx:.0f},{cy:.0f}) R={R:.0f}  angle={theta:.1f} deg")

        return {'angle_avg': round(theta, 2), 'angle_left': round(theta, 2),
                'angle_right': round(theta, 2),
                'cx': float(cx), 'cy': float(cy), 'R': float(R), 'valid': True}
    except Exception as e:
        if debug:
            print(f"  Circle: failed ({e})")
        return {'angle_avg': 90.0, 'angle_left': 90.0, 'angle_right': 90.0,
                'cx': 0, 'cy': 0, 'R': 0, 'valid': False}


def calc_poly(valid_contour, debug=True):
    """Polynomial tangent backup."""
    cy = valid_contour[:, 0]
    cx = valid_contour[:, 1]
    il, ir = np.argmin(cx), np.argmax(cx)

    def _pa(cpx, cpy, side_str, win=50):
        d = np.sqrt((cx - cpx) ** 2 + (cy - cpy) ** 2)
        near = d < win
        if np.sum(near) < 8:
            return 0.0
        xs = cx[near].astype(np.float64)
        ys = cy[near].astype(np.float64)
        _, ui = np.unique(np.round(ys), return_index=True)
        if len(ui) < 5:
            return 0.0
        xs, ys = xs[ui], ys[ui]
        ym = np.mean(ys)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            a, b, _ = np.polyfit(ys - ym, xs, 2)
        dx_dy = 2 * a * (cpy - ym) + b
        tv = np.array([dx_dy, -1.0])
        tn = np.linalg.norm(tv)
        if tn < 1e-10:
            return 0.0
        bv = np.array([-1.0, 0.0]) if side_str == 'left' else np.array([1.0, 0.0])
        return float(np.degrees(np.arccos(np.clip(np.dot(tv, bv) / tn, -1, 1))))

    al = _pa(cx[il], cy[il], 'left')
    ar = _pa(cx[ir], cy[ir], 'right')
    aa = (al + ar) / 2.0
    if debug:
        print(f"  Poly: L={al:.2f}  R={ar:.2f}  Avg={aa:.2f} deg")
    return {'angle_avg': round(aa, 2), 'angle_left': round(al, 2),
            'angle_right': round(ar, 2)}


# ============================================================
#  Visualization
# ============================================================

def visualize(img, roi, enhanced_roi, contour, valid_contour, baseline_y,
              yl_res, circle_res, poly_res, cx, cy, r, roi_bds, out_path):
    """2x3 diagnostic figure."""
    x1, y1, x2, y2 = roi_bds
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Contact Angle Auto-Measurement", fontsize=15, fontweight='bold')

    # (0,0) Full image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].add_patch(plt.Circle((cx, cy), r, fill=False, ec='red', lw=2))
    axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, ec='yellow', lw=1.5))
    axes[0, 0].set_title(f"Detection: ({cx},{cy}) r={r}", fontsize=10)
    axes[0, 0].axis('off')

    # (0,1) ROI
    axes[0, 1].imshow(cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"ROI Enhanced\n{enhanced_roi.shape[1]}x{enhanced_roi.shape[0]}px",
                         fontsize=10)
    axes[0, 1].axis('off')

    # (0,2) Segmentation
    if len(valid_contour) > 0:
        axes[0, 2].imshow(cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB))
        axes[0, 2].plot(contour[:, 1], contour[:, 0], 'r-', lw=1, alpha=0.5)
        axes[0, 2].plot(valid_contour[:, 1], valid_contour[:, 0], 'lime', lw=1.5)
        axes[0, 2].axhline(y=baseline_y, color='cyan', ls='--', lw=2)
        axes[0, 2].set_title(f"Segmentation\nBaseline y={baseline_y}", fontsize=10)
    axes[0, 2].axis('off')

    # (1,0) Y-L
    ax_yl = axes[1, 0]
    if yl_res.get('valid') and len(valid_contour) > 0:
        cz_v = baseline_y - valid_contour[:, 0]
        ax_yl.scatter(valid_contour[:, 1], cz_v, c='steelblue', s=5, alpha=0.5)
        ax_yl.axhline(y=0, color='red', ls='--', lw=1)
        if yl_res.get('tip'):
            ax_yl.scatter([yl_res['tip'][0]], [yl_res['tip'][1]],
                          c='red', s=100, marker='*', zorder=5)
        ax_yl.set_title(f"Y-L Fit\nL={yl_res['angle_left']:.1f}  R={yl_res['angle_right']:.1f} deg",
                        fontsize=10)
        ax_yl.set_xlabel('x (px)'); ax_yl.set_ylabel('z (px)')
        ax_yl.grid(True, alpha=0.3); ax_yl.set_aspect('equal')
    else:
        ax_yl.text(0.5, 0.5, 'Y-L skipped', ha='center', va='center',
                   transform=ax_yl.transAxes)

    # (1,1) Circle
    ax_c = axes[1, 1]
    if circle_res.get('valid') and len(valid_contour) > 0:
        cz_c = baseline_y - valid_contour[:, 0]
        ax_c.scatter(valid_contour[:, 1], cz_c, c='steelblue', s=5, alpha=0.5)
        ax_c.axhline(y=0, color='red', ls='--', lw=1)
        th = np.linspace(0, 2 * np.pi, 300)
        ax_c.plot(circle_res['cx'] + circle_res['R'] * np.cos(th),
                  baseline_y - (circle_res['cy'] + circle_res['R'] * np.sin(th)),
                  'orange', lw=2, alpha=0.7)
        ax_c.set_title(f"Circle Fit\nAngle={circle_res['angle_avg']:.1f} deg", fontsize=10)
        ax_c.set_xlabel('x (px)'); ax_c.set_ylabel('z (px)')
        ax_c.grid(True, alpha=0.3); ax_c.set_aspect('equal')
    else:
        ax_c.text(0.5, 0.5, 'Circle N/A', ha='center', va='center',
                  transform=ax_c.transAxes)

    # (1,2) Summary
    ax_s = axes[1, 2]
    ax_s.axis('off'); ax_s.set_xlim(0, 10); ax_s.set_ylim(0, 10)

    final = yl_res['angle_avg'] if yl_res.get('valid') else circle_res['angle_avg']
    if final < 90:
        cls_t, clr = "Hydrophilic", '#2563eb'
    elif final < 150:
        cls_t, clr = "Hydrophobic", '#f59e0b'
    else:
        cls_t, clr = "Superhydrophobic", '#ef4444'

    yp = 9.5
    ax_s.text(0.5, yp, "RESULTS SUMMARY", fontsize=13, fontweight='bold', color='#1e3a5f')
    yp -= 1.2
    items = [
        ("Method", yl_res.get('method', 'Circle')),
        ("", ""),
        ("Y-L Left", f"{yl_res['angle_left']:.2f} deg"),
        ("Y-L Right", f"{yl_res['angle_right']:.2f} deg"),
        ("Circle", f"{circle_res['angle_avg']:.2f} deg"),
        ("-" * 24, "-" * 10),
        ("* FINAL", f"{final:.2f} deg"),
        ("", ""),
        ("Poly L/R", f"{poly_res['angle_left']:.1f} / {poly_res['angle_right']:.1f} deg"),
    ]
    for lbl, val in items:
        if lbl.startswith("-"):
            ax_s.text(0.5, yp, lbl, fontsize=8, color='gray')
        elif lbl == "* FINAL":
            ax_s.text(0.5, yp, lbl, fontsize=12, fontweight='bold', color='#2563eb')
            ax_s.text(5.5, yp, val, fontsize=12, fontweight='bold', color='#2563eb')
        elif lbl == "Method":
            ax_s.text(0.5, yp, f"{lbl}:", fontsize=10, color='#333')
            ax_s.text(5.5, yp, val, fontsize=10, color='#059669', fontweight='bold')
        else:
            ax_s.text(0.5, yp, f"{lbl}:", fontsize=9, color='#555')
            ax_s.text(5.5, yp, val, fontsize=9, color='#333')
        yp -= 0.6
    ax_s.text(0.5, yp - 0.3, cls_t, fontsize=12, fontweight='bold', color=clr)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n  Result saved to: {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
#  Main pipeline
# ============================================================

def analyze_contact_angle(image_path, output_path=None, debug=True):
    """Full auto contact angle measurement."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if debug:
        print("=" * 60)
        print(f"  Contact Angle Auto-Measurement")
        print(f"  Image: {image_path}  |  {img.shape[1]}x{img.shape[0]}px")
        print("=" * 60)

    # 1. Detect
    if debug:
        print("\n[1/5] Detecting droplet...")
    candidates = find_candidates(gray, debug=debug)
    cx, cy, r = select_droplet(gray, candidates, debug=debug)

    # 2. ROI
    if debug:
        print(f"\n[2/5] Extracting ROI around ({cx},{cy}) r={r}...")
    margin = int(r * 2.0)
    x1, y1 = max(0, cx - margin), max(0, cy - margin)
    x2, y2 = min(img.shape[1], cx + margin), min(img.shape[0], cy + margin)
    roi = img[y1:y2, x1:x2]
    enhanced_roi = clahe_enhance(roi)
    if debug:
        print(f"  ROI: {roi.shape[1]}x{roi.shape[0]}px")

    # 3. Segment
    if debug:
        print(f"\n[3/5] Segmenting droplet...")
    contour, baseline_y, valid_contour = segment_in_roi(enhanced_roi, r, debug=debug)

    if len(valid_contour) < 10:
        raise ValueError("Too few valid contour points after segmentation.")

    # 4. Calculate
    if debug:
        print(f"\n[4/5] Computing contact angle...")
    yl_res = calc_yl(valid_contour, baseline_y, r, debug=debug)
    circle_res = calc_circle(valid_contour, baseline_y, debug=debug)
    poly_res = calc_poly(valid_contour, debug=debug)

    # Final
    final_angle = yl_res['angle_avg'] if yl_res.get('valid') else circle_res['angle_avg']
    cls_name = ("Hydrophilic" if final_angle < 90 else
                "Hydrophobic" if final_angle < 150 else "Superhydrophobic")

    # 5. Visualize
    if debug:
        print(f"\n[5/5] Generating figure...")
    visualize(img, roi, enhanced_roi, contour, valid_contour, baseline_y,
              yl_res, circle_res, poly_res, cx, cy, r,
              (x1, y1, x2, y2), output_path)

    if debug:
        print("\n" + "=" * 60)
        print(f"  Circle:   {circle_res['angle_avg']:.2f} deg")
        print(f"  Y-L:      {yl_res['angle_avg']:.2f} deg")
        print(f"  Poly:     {poly_res['angle_avg']:.2f} deg")
        print(f"  ---")
        print(f"  FINAL:    {final_angle:.2f} deg  ({cls_name})")
        print("=" * 60)

    return {
        'contact_angle': final_angle,
        'angle_left': yl_res['angle_left'],
        'angle_right': yl_res['angle_right'],
        'circle_angle': circle_res['angle_avg'],
        'poly_angle': poly_res['angle_avg'],
        'method': yl_res.get('method', 'Circle'),
        'baseline_y': int(baseline_y),
        'contour_points': len(valid_contour),
        'classification': cls_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Contact Angle Auto-Measurement")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output", "-o", default=None,
                        help="Output result image (default: <input>_result.jpg)")
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.splitext(args.image)[0] + "_result.jpg"

    try:
        r = analyze_contact_angle(args.image, output_path=args.output, debug=True)
        print(f"\nDone. Contact angle = {r['contact_angle']:.2f} deg "
              f"({r['classification']})")
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
