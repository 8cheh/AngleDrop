"""Substrate detection, drop segmentation and sub-pixel profile extraction.

Two deliberate departures from OpenDrop, both of which matter on front-lit
images where the stage below the contact line is bright:

  * the substrate line is found automatically (OpenDrop makes the user drag it),
  * profile points are refined to sub-pixel along the edge normal (OpenDrop
    keeps integer pixel coordinates).

The substrate is located *jointly* with the drop: a per-column edge profile
yields a handful of candidate stage edges, each is scored by how well a drop
segments against it, and the best candidate + orientation wins. This handles
both upright and inverted (captive/pendant) setups and images whose brightest
horizontal edge is not the substrate.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .geometry import Line

__all__ = ['read_image', 'enhance', 'detect_baseline', 'drop_mask',
           'profile_points', 'find_substrate']


def read_image(path: str) -> Optional[np.ndarray]:
    """Read a BGR image, tolerating non-ASCII paths on Windows."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def enhance(bgr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Edge-preserving denoise + CLAHE. Helps low-contrast frames, harmless otherwise."""
    den = cv2.bilateralFilter(bgr, 7, 50, 50)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------- baseline
def _refine_line(gray: np.ndarray, row: float, polarity: int, seed: int,
                 win: int = 40, max_slope: float = 0.12,
                 sob: Optional[np.ndarray] = None) -> Optional[Tuple[float, float]]:
    """Fit a near-horizontal stage edge near `row` with the given dark->bright sign.

    polarity=+1: dark above / bright below (upright drop). -1 is the reverse.
    """
    h, w = gray.shape
    if sob is None:
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        sob = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    sob = sob * polarity

    y0 = int(max(0, row - win)); y1 = int(min(h, row + win))
    if y1 - y0 < 10:
        return None
    band = sob[y0:y1, :]
    xs = np.arange(0, w, 3)
    sub = band[:, xs]
    cols = np.arange(sub.shape[1])
    idx = np.argmax(sub, axis=0)
    peak = sub[idx, cols]

    a = sub[np.clip(idx - 1, 0, None), cols]
    c = sub[np.clip(idx + 1, None, sub.shape[0] - 1), cols]
    den = a - 2 * peak + c
    off = np.where(np.abs(den) > 1e-6,
                   0.5 * (a - c) / np.where(np.abs(den) > 1e-6, den, 1.0), 0.0)
    ys = y0 + idx + np.clip(off, -1, 1)

    strong = peak > max(8.0, float(np.percentile(peak, 50)))
    X, Y = xs[strong].astype(float), ys[strong]
    if X.size < 15:
        X, Y = xs.astype(float), ys
    if X.size < 4:
        return None

    rng = np.random.default_rng(seed)
    best_in, best = None, None
    n = X.size
    for _ in range(200):
        i, j = rng.integers(0, n, 2)
        if abs(X[i] - X[j]) < w * 0.2:
            continue
        k = (Y[j] - Y[i]) / (X[j] - X[i])
        if abs(k) > max_slope:
            continue
        b = Y[i] - k * X[i]
        inl = np.abs(Y - (k * X + b)) < 3.0
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best = inl, (float(k), float(b))
    if best_in is None or best_in.sum() < 10:
        return None
    k, b = np.polyfit(X[best_in], Y[best_in], 1)
    if abs(k) > max_slope:
        return None
    return float(k), float(b)


def _candidate_rows(gray: np.ndarray, max_cands: int = 8) -> np.ndarray:
    """Distinct rows of strong horizontal dark->bright transitions, best first."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    sob = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    prof = sob.sum(axis=1)
    prof = np.convolve(prof, np.ones(15) / 15.0, mode='same')
    # local maxima, spaced apart
    order = np.argsort(prof)[::-1]
    rows = []
    for r in order:
        if not rows or all(abs(r - q) > 30 for q in rows):
            rows.append(int(r))
        if len(rows) >= max_cands:
            break
    return np.array(rows, dtype=int)


def detect_baseline(gray: np.ndarray, *, max_cands: int = 8,
                    seed: int = 0) -> Tuple[float, float]:
    """Best-effort single-line substrate detector (upright only)."""
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    for row in _candidate_rows(g, max_cands):
        fit = _refine_line(g, row, +1, seed)
        if fit is not None:
            return fit
    h, w = g.shape
    return 0.0, float(h * 0.75)


# ----------------------------------------------------------------- segmentation
def _grad_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude, smoothed just enough to be a stable edge measure."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _edge_reference(mag: np.ndarray) -> float:
    """Typical strength of a genuine edge in this frame.

    Used to turn the raw boundary gradient into a dimensionless 0..1 quality,
    so the segmentation score does not inherit the exposure of the image.
    """
    return max(1e-6, float(np.percentile(mag, 99)))


def _blob_quality(comp: np.ndarray, mag: np.ndarray, ref: float) -> Tuple[float, float]:
    """Convexity and boundary-edge strength of one candidate blob.

    A sessile drop is a convex body bounded by a real optical interface, so it
    is both compact (solidity close to 1) and outlined by a strong edge. The
    background texture that a brightness-only score happily accepts is neither:
    on the shipped example image the true drop scores solidity 0.79 / boundary
    gradient 161 while the texture blob that used to win scores 0.49 / 16.
    """
    m8 = comp.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    area = float(comp.sum())
    solidity = area / hull if hull > 1e-9 else 0.0

    er = cv2.erode(m8, np.ones((3, 3), np.uint8))
    boundary = (m8 > 0) & (er == 0)
    bgrad = float(mag[boundary].mean()) if boundary.any() else 0.0
    return solidity, min(1.0, bgrad / ref)


def _segment(gray: np.ndarray, line: Line, clearance: float = 3.0,
             min_area: int = 400, ref_grad: Optional[float] = None,
             mag: Optional[np.ndarray] = None):
    """Segment the drop sitting on `line`. Returns (mask, polarity, score)."""
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    zmap = line.z_map((h, w))
    inside = zmap > clearance

    if ref_grad is None or mag is None:
        mag = _grad_magnitude(gray) if mag is None else mag
        ref_grad = _edge_reference(mag) if ref_grad is None else ref_grad

    vals = g[inside]
    if vals.size < 500:
        return None, 0, -1.0
    thr, _ = cv2.threshold(vals.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best = None
    for polarity in (1, -1):
        m = ((g >= thr) if polarity > 0 else (g <= thr)) & inside
        m = (m * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x0, y0, bw, bh, area = stats[i]
            if area < min_area or bh < 8:
                continue
            if bw > 0.75 * w:
                # a real sessile drop does not span three quarters of the frame;
                # this is a stage/background sliver, not the drop
                continue
            if x0 <= 1 or x0 + bw >= w - 1 or y0 <= 1:
                # touching the frame is a background/reflection region, never a
                # centred sessile drop
                continue
            # centroid must be well inside the liquid side of the substrate
            cy = y0 + bh / 2.0
            col = x0 + bw / 2.0
            if zmap[int(np.clip(cy, 0, h - 1)), int(np.clip(col, 0, w - 1))] < clearance:
                continue
            # the blob must physically rest ON the substrate, not float above it
            yy0, yy1 = max(0, y0 - 2), min(h, y0 + bh + 2)
            xx0, xx1 = max(0, x0), min(w, x0 + bw)
            if zmap[yy0:yy1, xx0:xx1][(lab[yy0:yy1, xx0:xx1] == i)].min() > clearance + 4.0:
                continue
            # A drop is a compact, convex blob outlined by a strong edge.
            # Scoring on area alone (the previous behaviour) lets a large
            # low-contrast background region outrank the actual drop, which is
            # how the shipped example image used to be measured completely
            # wrong while still reporting ok=True.
            solidity, edge_q = _blob_quality(lab == i, mag, ref_grad)
            ar = bw / max(bh, 1)
            score = float(area) * solidity * edge_q
            score /= (1.0 + max(0.0, ar - 2.0))
            score *= 1.0 - min(1.0, max(0.0, (bw - 0.7 * w) / (0.3 * w)))
            if best is None or score > best[0]:
                best = (score, (lab == i).astype(np.uint8) * 255, polarity)
    if best is None:
        return None, 0, -1.0
    return best[1], best[2], best[0]


def drop_mask(gray: np.ndarray, line: Line, **kw) -> Tuple[Optional[np.ndarray], int]:
    """Segment the drop above (or below, for inverted) the given substrate."""
    m, pol, _ = _segment(gray, line, **kw)
    return m, pol


def find_substrate(gray: np.ndarray, *, max_cands: int = 8, seed: int = 0):
    """Jointly locate the substrate line, its orientation and the drop mask.

    Returns (Line, mask, polarity, score) or None.
    """
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    sob = cv2.Sobel(cv2.GaussianBlur(g, (5, 5), 0), cv2.CV_32F, 0, 1, ksize=5)
    # computed once and shared by every candidate: it is a per-frame constant
    mag = _grad_magnitude(g)
    ref_grad = _edge_reference(mag)
    best = None
    for row in _candidate_rows(g, max_cands):
        for polarity in (+1, -1):
            fit = _refine_line(g, row, polarity, seed, sob=sob)
            if fit is None:
                continue
            k, b = fit
            try:
                line = Line(k, b, w, invert=(polarity < 0))
            except ValueError:
                continue
            mask, mpol, score = _segment(g, line, ref_grad=ref_grad, mag=mag)
            if mask is None:
                continue
            if best is None or score > best[0]:
                best = (score, line, mask, mpol)
    if best is None:
        k, b = detect_baseline(g, max_cands=max_cands, seed=seed)
        line = Line(k, b, w, invert=False)
        mask, mpol, _ = _segment(g, line, ref_grad=ref_grad, mag=mag)
        if mask is None:
            return None
        best = (0.0, line, mask, mpol)
    return best[1], best[2], best[3], best[0]


# ----------------------------------------------------------------- profile points
def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def profile_points(gray: np.ndarray, mask: np.ndarray, line: Line,
                   grad_frac: float = 0.18, grad_min: float = 6.0) -> np.ndarray:
    """Left/right extreme mask pixel per row, sub-pixel refined. Returns (2, N).

    The gradient filter is load-bearing: the mask is cut off flat just above the
    substrate, and those cut pixels sit inside a uniform bright region. Left in,
    they drag the near-contact fit towards vertical.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.empty((2, 0))
    order = np.argsort(ys, kind='stable')
    ys, xs = ys[order], xs[order]
    bounds = np.searchsorted(ys, np.unique(ys), side='left')
    bounds = np.append(bounds, ys.size)
    pts = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        row = xs[s:e]
        y = ys[s]
        pts.append((row.min(), y))
        pts.append((row.max(), y))
    P = np.array(pts, dtype=float).T

    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    x, y = P[0], P[1]
    m0 = _bilinear(mag, x, y)
    keep = m0 > max(grad_min, grad_frac * float(np.percentile(m0, 90)))
    if keep.sum() >= 20:
        P = P[:, keep]
        x, y, m0 = P[0], P[1], m0[keep]

    nx, ny = _bilinear(gx, x, y), _bilinear(gy, x, y)
    nn = np.hypot(nx, ny) + 1e-9
    nx, ny = nx / nn, ny / nn
    mm = _bilinear(mag, x - nx, y - ny)
    mp = _bilinear(mag, x + nx, y + ny)
    den = mm - 2 * m0 + mp
    t = np.where(np.abs(den) > 1e-6,
                 0.5 * (mm - mp) / np.where(np.abs(den) > 1e-6, den, 1.0), 0.0)
    t = np.clip(t, -1, 1)
    return np.stack([x + t * nx, y + t * ny])
