"""End-to-end measurement pipeline."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from . import fitting as F
from .detect import drop_mask, enhance, find_substrate, profile_points, read_image
from .geometry import Line
from .superres import super_resolve

__all__ = ['SideFit', 'MeasureResult', 'measure', 'measure_roi', 'measure_file', 'METHODS']

METHODS = ('auto', 'circle', 'ellipse', 'poly', 'line')


@dataclass
class SideFit:
    ok: bool = False
    method: str = ''
    angle_deg: Optional[float] = None
    rms_px: float = float('inf')
    n_points: int = 0
    contact_xy: Optional[Tuple[float, float]] = None
    curve_xy: Optional[np.ndarray] = None      # (2, M) fitted profile, image coords
    tangent_xy: Optional[np.ndarray] = None    # (2, 2) tangent segment, image coords

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ok': self.ok, 'method': self.method,
            'angle_deg': None if self.angle_deg is None else round(self.angle_deg, 2),
            'rms_px': None if not np.isfinite(self.rms_px) else round(self.rms_px, 3),
            'n_points': self.n_points,
            'contact_xy': None if self.contact_xy is None
            else [round(float(v), 1) for v in self.contact_xy],
        }


@dataclass
class MeasureResult:
    filename: str = ''
    ok: bool = False
    error: Optional[str] = None
    method: str = ''
    left: SideFit = field(default_factory=SideFit)
    right: SideFit = field(default_factory=SideFit)
    avg_angle: Optional[float] = None
    asymmetry: Optional[float] = None
    baseline_k: float = 0.0
    baseline_b: float = 0.0
    baseline_tilt_deg: float = 0.0
    base_width_px: Optional[float] = None
    height_px: Optional[float] = None
    n_profile: int = 0
    confidence: float = 0.0
    profile_xy: Optional[np.ndarray] = None
    image_size: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'filename': self.filename, 'ok': self.ok, 'error': self.error,
            'method': self.method,
            'left': self.left.to_dict(), 'right': self.right.to_dict(),
            'avg_angle': None if self.avg_angle is None else round(self.avg_angle, 2),
            'asymmetry': None if self.asymmetry is None else round(self.asymmetry, 2),
            'baseline': {'k': round(self.baseline_k, 6),
                         'b': round(self.baseline_b, 2),
                         'tilt_deg': round(self.baseline_tilt_deg, 3)},
            'base_width_px': None if self.base_width_px is None
            else round(self.base_width_px, 1),
            'height_px': None if self.height_px is None else round(self.height_px, 1),
            'n_profile': self.n_profile,
            'confidence': round(self.confidence, 3),
            'image_size': list(self.image_size),
        }


def _fit_side(sel: np.ndarray, contact: np.ndarray, base_w: float, side: str,
              method: str, line: Line) -> SideFit:
    """Fit one side. `sel` is (2, N) baseline coords for that half of the drop."""
    out = SideFit()
    S = max(base_w, 20.0)                     # uniform scale -> angles unchanged
    r, z = sel[0] / S, sel[1] / S
    cr = contact[0] / S
    out.n_points = int(r.size)
    if r.size < 5:
        return out

    chosen: Optional[Tuple[str, float, float, float]] = None   # name, ang, r0, rms
    curve = None

    def circle_try():
        fit = F.fit_circle(r, z)
        if fit is None:
            return None
        rc, zc, R, rms = fit
        ang, r0 = F.circle_contact(rc, zc, R, side, near_r=cr)
        if ang is None:
            return None
        return ('circle', ang, r0, rms, ('circle', rc, zc, R))

    def line_try():
        fit = F.fit_line(r, z)
        if fit is None:
            return None
        m, c, mx = fit
        ang, r0 = F.line_contact(m, c, side)
        if ang is None:
            return None
        resid = float(np.sqrt(np.mean((np.polyval([m, c], z) - r) ** 2)))
        return ('line', ang, r0, resid, ('poly', np.array([m, c])), mx)

    if method == 'circle':
        chosen = circle_try()
    elif method == 'line':
        chosen = line_try()
    elif method == 'ellipse':
        co = F.fit_ellipse(r, z)
        if co is not None:
            ang, r0 = F.conic_contact(co, side, near_r=cr)
            if ang is not None:
                chosen = ('ellipse', ang, r0, F.conic_rms(co, r, z), ('conic', co))
    elif method == 'poly':
        fit = F.fit_poly(r, z, 2)
        if fit is not None:
            co, r0, slope, rms = fit
            ang, r0 = F.poly_contact(co, side)
            if ang is not None:
                chosen = ('poly', ang, r0, rms, ('poly', co))
    else:                                      # auto: straight -> line, else circle
        lt = line_try()
        if lt is not None and lt[5] * S < 1.0:
            chosen = lt[:5]
        else:
            chosen = circle_try() or (lt[:5] if lt else None)

    if chosen is None:
        return out
    name, ang, r0, rms, model = chosen[0], chosen[1], chosen[2], chosen[3], chosen[4]
    if not (0.0 < ang < 180.0):
        return out

    # sample the fitted curve for display, from the contact up to the data top
    z_hi = float(z.max())
    if model[0] == 'circle':
        _, rc, zc, R = model
        if abs(zc) < R:
            phi0 = math.atan2(0 - zc, r0 - rc)
            z_top = min(z_hi, zc + R)
            phi1 = math.asin(max(-1.0, min(1.0, (z_top - zc) / R)))
            phi1 = phi1 if abs(phi1 - phi0) < math.pi else phi0
            phis = np.linspace(phi0, phi1, 60)
            curve = np.stack([rc + R * np.cos(phis), zc + R * np.sin(phis)])
    elif model[0] == 'poly':
        zs = np.linspace(0, z_hi, 60)
        curve = np.stack([np.polyval(model[1], zs), zs])
    elif model[0] == 'conic':
        co = model[1]
        zs = np.linspace(0, z_hi, 60)
        A, B, C, D, E, Fc = co
        aa = A
        bb = B * zs + D
        cc = C * zs * zs + E * zs + Fc
        disc = bb * bb - 4 * aa * cc
        good = disc >= 0
        if good.sum() >= 2:
            sq = np.sqrt(np.maximum(disc, 0))
            r1 = (-bb - sq) / (2 * aa)
            r2 = (-bb + sq) / (2 * aa)
            pick = np.where(np.abs(r1 - r0) < np.abs(r2 - r0), r1, r2)
            curve = np.stack([pick[good], zs[good]])

    out.ok = True
    out.method = name
    out.angle_deg = float(ang)
    out.rms_px = float(rms * S)
    contact_rz = np.array([r0 * S, 0.0])
    out.contact_xy = tuple(float(v) for v in line.to_xy(contact_rz))
    if curve is not None and curve.shape[1] >= 2:
        out.curve_xy = line.to_xy(curve * S)
    # tangent segment for display
    L = 0.22 * base_w
    th = math.radians(ang)
    dr = math.cos(th) * (1 if side == 'left' else -1)
    tip = contact_rz + np.array([dr * L, math.sin(th) * L])
    out.tangent_xy = line.to_xy(np.stack([contact_rz, tip], axis=1))
    return out


def measure(bgr: np.ndarray, filename: str = '', *, method: str = 'auto',
            win_frac: float = 0.25, use_enhance: bool = False,
            use_sr: bool = False, sr_scale: float = 3.0,
            baseline: Optional[Tuple[float, float]] = None) -> MeasureResult:
    """Measure the contact angle of a single sessile-drop image.

    ``use_sr`` runs Real-ESRGAN super-resolution as the first preprocessing
    step (before enhancement); ``sr_scale`` is the requested upscale factor.
    """
    res = MeasureResult(filename=filename, method=method)
    if bgr is None:
        res.error = 'image could not be read'
        return res
    if method not in METHODS:
        res.error = f'unknown method {method!r}'
        return res

    if use_sr:
        try:
            bgr = super_resolve(bgr, scale=sr_scale)
        except Exception as e:  # model missing / onnxruntime absent -> fail loudly
            res.error = f'super-resolve failed: {e}'
            return res

    h, w = bgr.shape[:2]
    res.image_size = (int(w), int(h))
    work = enhance(bgr) if use_enhance else bgr
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    if baseline is not None:
        k, b = baseline
        ln = Line(k, b, w, invert=False)
        mask, _ = drop_mask(gray, ln)
    else:
        found = find_substrate(gray)
        if found is None:
            res.error = 'no drop found'
            return res
        ln, mask, _, _ = found
    res.baseline_k, res.baseline_b = float(ln.k), float(ln.b)
    res.baseline_tilt_deg = ln.angle_deg
    if mask is None:
        res.error = 'no drop found above the substrate'
        return res

    P = profile_points(gray, mask, ln)
    if P.shape[1] < 20:
        res.error = f'too few profile points ({P.shape[1]})'
        return res
    rz = ln.to_rz(P)
    rz = rz[:, rz[1] > 0.5]
    if rz.shape[1] < 20:
        res.error = 'drop profile is not above the substrate'
        return res
    res.profile_xy = ln.to_xy(rz)
    res.n_profile = int(rz.shape[1])
    res.height_px = float(rz[1].max())

    mid = 0.5 * (rz[0].min() + rz[0].max())
    halves = {'left': rz[:, rz[0] < mid], 'right': rz[:, rz[0] >= mid]}
    if min(v.shape[1] for v in halves.values()) < 6:
        res.error = 'drop profile is one-sided'
        return res

    # contact point = OUTERMOST point of the lowest band, not simply argmin(z):
    # the mask is cut flat above the substrate so many points tie on z.
    zband = rz[1].min() + max(3.0, 0.06 * res.height_px)
    contacts = {}
    for s, v in halves.items():
        low = v[:, v[1] <= zband]
        if low.shape[1] == 0:
            low = v[:, [int(np.argmin(v[1]))]]
        j = int(np.argmin(low[0])) if s == 'left' else int(np.argmax(low[0]))
        contacts[s] = low[:, j]
    base_w = float(contacts['right'][0] - contacts['left'][0])
    if base_w <= 5:
        res.error = 'degenerate base width'
        return res
    res.base_width_px = base_w

    for s in ('left', 'right'):
        v = halves[s]
        d = np.hypot(v[0] - contacts[s][0], v[1] - contacts[s][1])
        m = d < win_frac * base_w
        if m.sum() < 8:
            m = d <= np.percentile(d, 40)
        fit = _fit_side(v[:, m], contacts[s], base_w, s, method, ln)
        setattr(res, s, fit)

    vals = [f.angle_deg for f in (res.left, res.right) if f.ok and f.angle_deg]
    if not vals:
        res.error = 'both sides failed to fit'
        return res
    res.ok = True
    res.avg_angle = float(np.mean(vals))
    if res.left.ok and res.right.ok:
        res.asymmetry = abs(res.left.angle_deg - res.right.angle_deg)

    # confidence: fit quality relative to drop size, point count, L/R agreement
    conf = 1.0
    for f in (res.left, res.right):
        if f.ok:
            conf *= math.exp(-f.rms_px / max(0.02 * base_w, 2.0))
            conf *= min(1.0, f.n_points / 25.0)
        else:
            conf *= 0.35
    if res.asymmetry is not None:
        conf *= math.exp(-res.asymmetry / 25.0)
    res.confidence = float(max(0.0, min(1.0, conf)))
    return res


def measure_file(path: str, **kw) -> MeasureResult:
    img = read_image(path)
    res = measure(img, os.path.basename(path), **kw)
    if img is None:
        res.error = 'image could not be read'
    return res


def measure_roi(bgr: np.ndarray, roi, filename: str = '', **kw) -> MeasureResult:
    """Measure within a user-selected rectangle, mapping results back to full coords.

    `roi` is (x, y, w, h) in full-image pixel coordinates. The pipeline runs on
    the crop (so the substrate search is constrained to the selected region),
    then every image-space result is offset back to the original image.
    """
    h, w = bgr.shape[:2]
    x0 = int(max(0, round(roi[0])))
    y0 = int(max(0, round(roi[1])))
    x1 = int(min(w, round(roi[0] + roi[2])))
    y1 = int(min(h, round(roi[1] + roi[3])))
    if x1 - x0 < 20 or y1 - y0 < 20:
        res = MeasureResult(filename=filename)
        res.error = '所选区域太小'
        res.image_size = (int(w), int(h))
        return res

    crop = bgr[y0:y1, x0:x1]
    res = measure(crop, filename, **kw)
    res.image_size = (int(w), int(h))

    # shift the substrate intercept: y_full = k*x_full + (b_crop + y0 - k*x0)
    res.baseline_b = res.baseline_b + y0 - res.baseline_k * x0
    off = np.array([[x0], [y0]], dtype=float)

    if res.profile_xy is not None:
        res.profile_xy = res.profile_xy + off
    for side in (res.left, res.right):
        if side.contact_xy is not None:
            side.contact_xy = (side.contact_xy[0] + x0, side.contact_xy[1] + y0)
        if side.curve_xy is not None:
            side.curve_xy = side.curve_xy + off
        if side.tangent_xy is not None:
            side.tangent_xy = side.tangent_xy + off
    return res
