"""Pendant-drop tensiometry: fitting a Young-Laplace shape to a profile.

This module turns an extracted liquid/gas profile into a surface tension.  The
chain is

    profile (r, z) in image pixels
        -> fit (Bo, R0, apex_x, apex_y, rotation)          [young_laplace_fit]
        -> gamma = delta_rho * g * R0_metres**2 / Bo       [surface_tension]

Everything here works in *image* coordinates: ``x`` to the right, ``y``
downward, and the drop hanging with its apex at the largest ``y``.  The
mapping from the apex-centred Young-Laplace meridian to image coordinates is

    p(s, branch) = Rot(rotation) @ [branch * R0 * r(s), -R0 * z(s)] + apex

The minus sign on ``z`` is the whole coordinate convention in one character:
``z`` points *into the liquid* (upward for a pendant drop) while image ``y``
points down.  Getting it backwards produces a plausible-looking fit to a
mirrored drop, so ``tests/test_tensiometry.py`` checks it explicitly.

Two quantities are reported besides the fit itself, and both matter more than
the fit residual:

* ``shape_parameter`` (P_s) — how far the drop is from a sphere.  Surface
  tension lives in the *deviation* from sphericity, so a near-spherical drop
  has no surface-tension information in it no matter how good the camera is.
  Published critical values are 0.19-0.35 depending on the configuration.
* ``worthington`` (Wo) — how close the drop is to detaching.  Berry et al.
  introduced it precisely because the Bond number alone does not predict the
  achievable precision.

Clean-room implementation from the published mathematics; see
``cadrop/younglaplace.py`` for the equation references.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .fitting import fit_circle
from .younglaplace import YoungLaplaceShape

__all__ = [
    'PendantFitResult', 'synthesise_pendant_drop', 'detect_apex_and_radius',
    'young_laplace_fit', 'shape_parameter', 'shape_parameter_from_profile',
    'worthington_number', 'surface_tension', 'pixel_scale_from_needle',
    'GRAVITY', 'AIR_DENSITY',
]

#: Standard gravity, m/s^2.  Use a local value if you have one; the variation
#: across the Earth's surface is under 0.1%, which is far below the calibration
#: term in the error budget.
GRAVITY = 9.80665

#: Density of air at 25 C, 1 atm, kg/m^3.  Ignoring it inflates gamma by ~0.12%.
AIR_DENSITY = 1.184

#: Beyond this the profile is no longer a single-valued drop silhouette.
_MAX_ROTATION_DEG = 15.0


# --------------------------------------------------------------------- result
@dataclass
class PendantFitResult:
    ok: bool = False
    error: Optional[str] = None
    bond: Optional[float] = None
    radius_px: Optional[float] = None
    apex_x: Optional[float] = None
    apex_y: Optional[float] = None
    rotation_deg: Optional[float] = None
    rms_px: float = float('inf')
    n_points: int = 0
    shape_parameter: Optional[float] = None
    #: Dimensionless arc length covered by the fitted profile.
    arc_length: Optional[float] = None

    def to_dict(self) -> dict:
        def r(v, n=4):
            return None if v is None else round(float(v), n)
        return {
            'ok': self.ok, 'error': self.error,
            'bond': r(self.bond), 'radius_px': r(self.radius_px, 2),
            'apex': None if self.apex_x is None else [r(self.apex_x, 1), r(self.apex_y, 1)],
            'rotation_deg': r(self.rotation_deg, 3),
            'rms_px': None if not np.isfinite(self.rms_px) else r(self.rms_px, 3),
            'n_points': self.n_points,
            'shape_parameter': r(self.shape_parameter, 4),
            'arc_length': r(self.arc_length, 3),
        }


# ------------------------------------------------------------------ synthesis
def _place(shape: YoungLaplaceShape, s: np.ndarray, radius_px: float,
           apex: Tuple[float, float], rotation_deg: float) -> np.ndarray:
    """Map an apex-centred meridian onto image coordinates in both branches."""
    r, z, _ = shape.profile(s)
    w = math.radians(rotation_deg)
    cos_w, sin_w = math.cos(w), math.sin(w)

    local = np.stack([radius_px * r, -radius_px * z])          # (2, N), z up -> y down
    rot = np.array([[cos_w, -sin_w], [sin_w, cos_w]])
    plus = rot @ local + np.asarray(apex, dtype=float)[:, None]
    minus = rot @ (local * np.array([[-1.0], [1.0]])) + np.asarray(apex, dtype=float)[:, None]
    return np.concatenate([plus, minus], axis=1)


def synthesise_pendant_drop(bond: float, radius_px: float,
                            apex: Tuple[float, float],
                            rotation_deg: float = 0.0,
                            s_top: Optional[float] = None,
                            n_per_branch: int = 80,
                            noise_px: float = 0.0,
                            seed: int = 0) -> np.ndarray:
    """Generate a pendant-drop profile in image coordinates.

    Useful both as the forward model for tests and as a way to check what a
    given drop size and Bond number would actually look like before running an
    experiment.  Returns an ``(2, N)`` array of ``(x, y)`` pixels.
    """
    shape = YoungLaplaceShape(bond, invert=True)
    if s_top is None:
        # stop a little short of where the drop closes
        s_top = 0.92 * shape.s_max
    s = np.linspace(0.0, min(s_top, shape.s_max), n_per_branch)
    pts = _place(shape, s, radius_px, apex, rotation_deg)
    if noise_px:
        rng = np.random.default_rng(seed)
        pts = pts + rng.normal(0.0, noise_px, pts.shape)
    return pts


# ------------------------------------------------------------- initial guesses
def detect_apex_and_radius(pts: np.ndarray, cap_frac: float = 0.25
                           ) -> Tuple[Tuple[float, float], float]:
    """Estimate the apex position and radius of curvature from the bottom cap.

    A pendant drop's lowest cap is nearly circular, so an algebraic circle fit
    to the lowest ``cap_frac`` of the profile gives both the apex and R0 in one
    step.  The apex is taken as the lowest point on that circle rather than the
    lowest measured pixel, which is noise-sensitive.
    """
    x, y = pts[0], pts[1]
    if x.size < 6:
        raise ValueError('need at least 6 points to detect the apex')

    # "lowest" means largest y in image coordinates
    y_lo, y_hi = float(y.min()), float(y.max())
    cut = y_hi - cap_frac * max(y_hi - y_lo, 1e-9)
    sel = y >= cut
    if sel.sum() < 6:
        order = np.argsort(y)[::-1][:max(6, x.size // 4)]
        sel = np.zeros_like(y, dtype=bool)
        sel[order] = True

    xs, ys = x[sel], y[sel]
    a = np.column_stack([xs, ys, np.ones_like(xs)])
    sol, *_ = np.linalg.lstsq(a, xs ** 2 + ys ** 2, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx ** 2 + cy ** 2
    if not np.isfinite(r2) or r2 <= 0:
        raise ValueError('apex circle fit degenerated')
    radius = math.sqrt(r2)

    # apex = lowest point of the fitted circle
    return (float(cx), float(cy + radius)), float(radius)


# ----------------------------------------------------------------------- fit
def _residuals(pts: np.ndarray, shape: YoungLaplaceShape, radius_px: float,
               apex: Tuple[float, float], rotation_deg: float,
               dense: Optional[int] = None) -> np.ndarray:
    """Distance from every measured point to the nearest point on the model.

    Distances are unsigned.  The minimum of the sum of squares is attained
    when the model curve passes through the data, which is the geometric fit we
    want; a signed variant would only change the convergence path.

    The model is a polyline and the query returns the nearest *vertex*, so the
    vertex spacing is itself an error term: sampling the meridian every 0.015
    in arc length puts a 0.7 px floor under the residual at R0 = 150 px, which
    is larger than the precision the fit is supposed to reach.  The spacing is
    therefore tied to the on-screen scale, keeping it near a tenth of a pixel.
    """
    if dense is None:
        spacing = 0.15                                   # pixels between vertices
        dense = int(np.clip(shape.s_max * radius_px / spacing, 400, 20000))
    s = np.linspace(0.0, shape.s_max, dense)
    model = _place(shape, s, radius_px, apex, rotation_deg)
    tree = cKDTree(model.T)
    d, _ = tree.query(pts.T, k=1)
    return np.asarray(d, dtype=float)


def young_laplace_fit(pts: np.ndarray,
                      bond_grid: Optional[np.ndarray] = None,
                      bond_bounds: Tuple[float, float] = (1e-3, 20.0),
                      max_nfev: int = 120) -> PendantFitResult:
    """Fit a pendant-drop Young-Laplace shape to ``(x, y)`` profile points.

    Parameters are ``(Bo, R0_px, apex_x, apex_y, rotation_deg)``.  Fitting
    starts from a circle fit to the bottom cap, then a coarse sweep over the
    Bond number, then bounded least squares.  The sweep matters: the Bond
    number enters the objective only through the shape, so a local optimiser
    started far from the right branch can settle on a near-spherical solution
    with a superficially acceptable residual.
    """
    res = PendantFitResult()
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] != 2:
        res.error = 'profile must be a (2, N) array'
        return res
    if pts.shape[1] < 10:
        res.error = f'too few profile points ({pts.shape[1]})'
        return res
    if not np.all(np.isfinite(pts)):
        res.error = 'profile contains non-finite values'
        return res

    res.n_points = int(pts.shape[1])

    try:
        apex0, r0 = detect_apex_and_radius(pts)
    except ValueError as exc:
        res.error = f'apex detection failed: {exc}'
        return res

    lo, hi = bond_bounds
    grid = bond_grid if bond_grid is not None else np.geomspace(max(lo, 1e-3), min(hi, 5.0), 14)

    # --- stage 1: sweep Bond number with the apex and radius from the cap fit
    best_bo, best_cost = float(grid[0]), math.inf
    for bo in grid:
        try:
            sh = YoungLaplaceShape(float(bo), invert=True)
        except (RuntimeError, ValueError):
            continue
        if sh.s_max < 0.5:
            continue
        d = _residuals(pts, sh, r0, apex0, 0.0)
        cost = float(np.mean(d ** 2))
        if cost < best_cost:
            best_bo, best_cost = float(bo), cost

    # --- stage 2: bounded least squares over all five parameters
    shape_cache = {}

    def make_shape(bo: float) -> YoungLaplaceShape:
        key = round(float(bo), 9)
        if key not in shape_cache:
            shape_cache[key] = YoungLaplaceShape(float(bo), invert=True)
        return shape_cache[key]

    span = max(float(pts[0].max() - pts[0].min()),
               float(pts[1].max() - pts[1].min()), 1.0)

    def cost_vector(p):
        bo, radius, ax, ay, rot = p
        try:
            sh = make_shape(bo)
        except (RuntimeError, ValueError):
            return np.full(pts.shape[1], 1e6)
        if sh.s_max < 0.5:
            return np.full(pts.shape[1], 1e6)
        return _residuals(pts, sh, radius, (ax, ay), rot)

    p0 = np.array([best_bo, r0, apex0[0], apex0[1], 0.0], dtype=float)
    lower = np.array([max(lo, 1e-4), 1.0, pts[0].min() - span, pts[1].min() - span,
                      -_MAX_ROTATION_DEG])
    upper = np.array([hi, max(span * 20.0, 10.0), pts[0].max() + span,
                      pts[1].max() + span, _MAX_ROTATION_DEG])

    try:
        sol = least_squares(cost_vector, p0, bounds=(lower, upper),
                            method='trf', x_scale='jac', max_nfev=max_nfev,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
    except (ValueError, RuntimeError) as exc:
        res.error = f'least-squares failed: {exc}'
        return res

    bo, radius, ax, ay, rot = (float(v) for v in sol.x)
    resid = cost_vector(sol.x)

    res.ok = True
    res.bond = bo
    res.radius_px = radius
    res.apex_x, res.apex_y = ax, ay
    res.rotation_deg = rot
    res.rms_px = float(np.sqrt(np.mean(resid ** 2)))

    sh = make_shape(bo)
    # arc length actually covered by the data: project the furthest point
    far = np.argmax(np.hypot(pts[0] - ax, pts[1] - ay))
    res.arc_length = float(sh.closest(float(pts[0][far]), float(pts[1][far]))[0])
    res.shape_parameter = shape_parameter(sh, res.arc_length)
    return res


# ------------------------------------------------------- derived quantities
def shape_parameter_from_profile(profile: np.ndarray,
                                 apex_band: float = 0.05
                                 ) -> Tuple[Optional[float], Optional[float]]:
    """Shape parameter P_s measured straight from a profile, no Y-L fit.

    Returns ``(P_s, R0_px)``, or ``(None, None)`` if the profile is unusable.

    ``profile`` is ``(2, N)`` in baseline coordinates ``(r, z)`` with ``z``
    measured up from the substrate, i.e. exactly what ``cadrop.detect.
    profile_points`` + ``Line.to_rz`` already produce for a sessile drop.  The
    apex is the highest point.

    This exists so the conditioning question can be answered on data that
    already exists.  ``P_s`` is the fraction of the projected drop area lying
    outside the inscribed circle of radius equal to the apex radius of
    curvature: zero for a sphere, growing as gravity deforms the drop.  It is
    scale-free and liquid-independent, and published critical values for a
    0.1 mJ/m^2 error target sit at 0.19-0.35, so it is a cheap go/no-go test
    for whether a surface tension is recoverable at all.

    ``apex_band`` sets how much of the drop height is used for the apex circle
    fit.  It trades locality against noise tolerance: measured against exact
    Young-Laplace shapes, 0.05 of the height is within ~2% across Bo = 0.1-0.5
    while 0.25 of it is within ~12%.  Widen it only if the extracted profile is
    too noisy to support a tight cap fit.
    """
    pts = np.asarray(profile, dtype=float)
    if pts.ndim != 2 or pts.shape[0] != 2 or pts.shape[1] < 12:
        return None, None
    pts = pts[:, np.all(np.isfinite(pts), axis=0)]
    if pts.shape[1] < 12:
        return None, None

    r, z = pts[0], pts[1]
    z_lo, z_hi = float(z.min()), float(z.max())
    height = z_hi - z_lo
    # The apex must actually stand above the substrate.  A profile that is flat
    # or sits below its own baseline means the drop was mis-segmented, and
    # feeding that here produces a meaningless "shape parameter" rather than an
    # error -- which is the failure mode this whole diagnostic exists to avoid.
    if z_hi < 2.0 or height <= 1.0:
        return None, None

    # --- apex radius of curvature from a band just below the apex
    cap = z >= z_hi - apex_band * height
    if cap.sum() < 6:
        order = np.argsort(z)[::-1][:max(6, pts.shape[1] // 6)]
        cap = np.zeros_like(z, dtype=bool)
        cap[order] = True

    cr, cz = r[cap], z[cap]
    if np.unique(cr).size < 3:
        return None, None
    # Use the package's geometric fitter (algebraic start + Gauss-Newton): a
    # plain algebraic fit to a partial cap is biased towards too small a radius,
    # which is exactly the quantity P_s is measured against.
    scale = max(height, 1e-9)
    fit = fit_circle(cr / scale, cz / scale)
    if fit is None:
        return None, None
    _, _, r0, _ = fit
    r0 *= scale
    if not np.isfinite(r0) or not (0.02 * height < r0 < 50.0 * height):
        return None, None                     # implausible cap fit

    # --- projected (silhouette) area by the shoelace formula
    mid = 0.5 * (r.min() + r.max())
    order = np.argsort(z)
    rs, zs = r[order], z[order]
    right = rs >= mid
    if right.sum() < 3 or (~right).sum() < 3:
        return None, None
    poly = np.concatenate([
        np.stack([rs[right], zs[right]], axis=1),
        np.stack([rs[~right], zs[~right]], axis=1)[::-1],
    ])
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area <= 1e-9:
        return None, None

    ps = float(abs(area - math.pi * r0 * r0) / area)
    # P_s = 1 corresponds to a hemisphere; a real drop silhouette is always
    # well below that.  Anything above means the profile or the cap fit is
    # broken, so report nothing rather than an impressive-looking number.
    if not (0.0 <= ps <= 1.0):
        return None, None
    return ps, float(r0)


def shape_parameter(shape: YoungLaplaceShape, s_top: float) -> float:
    """Hajirahimi/Hoorfar shape parameter P_s over the measured extent.

    ``P_s`` is the fraction of the projected drop area that lies outside the
    inscribed circle of radius equal to the apex radius of curvature.  It is
    zero for a full sphere and is both scale-free and liquid-independent, which
    makes it the cheapest available go/no-go test for whether a surface tension
    can be determined from a given profile at all.  Published critical values
    (for an error target of 0.1 mJ/m^2) are 0.19-0.35 depending on the holder.
    """
    area = shape._quadrature(s_top, lambda r, z, phi: 2.0 * r * np.sin(phi))
    if area <= 1e-12:
        return 0.0
    return float(abs(area - math.pi) / area)


def pixel_scale_from_needle(needle_diameter_mm: float,
                            needle_diameter_px: float) -> float:
    """Millimetres per pixel, from a holder of known outer diameter.

    This is the only length reference in the measurement, and since
    ``gamma ~ scale**2`` its relative error is doubled into the result.
    """
    if needle_diameter_px <= 0:
        raise ValueError('needle_diameter_px must be positive')
    return float(needle_diameter_mm) / float(needle_diameter_px)


def surface_tension(delta_rho: float, radius_px: float, px_size_mm: float,
                    bond: float, gravity: float = GRAVITY) -> float:
    """Surface tension in mN/m from a fitted Bond number and apex radius.

        gamma = delta_rho * g * R0**2 / Bo

    ``radius_px`` is the apex radius of curvature in pixels and
    ``px_size_mm`` the calibration in millimetres per pixel.
    """
    if bond <= 0:
        raise ValueError('bond must be positive')
    r0_m = float(radius_px) * float(px_size_mm) / 1000.0
    return float(delta_rho * gravity * r0_m ** 2 / bond) * 1000.0


def worthington_number(delta_rho: float, volume_m3: float,
                       gamma_mN_m: float, needle_diameter_m: float,
                       gravity: float = GRAVITY) -> float:
    """Worthington number, ``Wo = delta_rho * g * V / (pi * gamma * D)``.

    This uses the Kratz & Kierfeld / Berry convention.  A second convention in
    the literature omits the ``pi`` and so differs by ``2*pi``; check which one
    a quoted threshold came from before comparing.
    """
    if gamma_mN_m <= 0 or needle_diameter_m <= 0:
        raise ValueError('gamma and needle diameter must be positive')
    return float(delta_rho * gravity * volume_m3 /
                 (math.pi * gamma_mN_m * 1e-3 * needle_diameter_m))
