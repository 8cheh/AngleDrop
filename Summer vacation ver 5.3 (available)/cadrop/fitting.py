"""Profile fits and contact-angle extraction.

All fits work in baseline coordinates (r, z) with the substrate at z = 0, and
are expected to be fed points from ONE side of the drop, restricted to a window
around the contact line.

The angle convention is the important part. For the right-hand contact point the
solid direction pointing into the drop is -r, for the left-hand one it is +r. So
with the profile tangent oriented upwards (tz > 0):

    theta_left  = atan2(tz,  tr)
    theta_right = atan2(tz, -tr)

Both land in (0, 180) degrees with no absolute values anywhere, which is what
lets hydrophobic drops (theta > 90) be represented at all.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = [
    'angle_from_tangent', 'fit_circle', 'fit_ellipse', 'fit_poly', 'fit_line',
    'circle_contact', 'conic_contact', 'poly_contact', 'line_contact',
]


def angle_from_tangent(tr: float, tz: float, side: str) -> Optional[float]:
    """Contact angle in degrees from a tangent vector, measured through the liquid."""
    if not (np.isfinite(tr) and np.isfinite(tz)):
        return None
    if tz < 0:                       # orient the tangent upwards
        tr, tz = -tr, -tz
    if tz <= 0:
        return None
    a = math.atan2(tz, tr if side == 'left' else -tr)
    return math.degrees(a)


# --------------------------------------------------------------------- circle
def fit_circle(r: np.ndarray, z: np.ndarray):
    """Algebraic (Kasa) start + geometric Gauss-Newton. Returns (rc, zc, R, rms)."""
    if r.size < 4:
        return None
    A = np.column_stack([r, z, np.ones_like(r)])
    rhs = r ** 2 + z ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    rc, zc = sol[0] / 2.0, sol[1] / 2.0
    R2 = sol[2] + rc ** 2 + zc ** 2
    if not np.isfinite(R2) or R2 <= 0:
        return None
    R = math.sqrt(R2)

    for _ in range(25):
        d = np.hypot(r - rc, z - zc)
        if np.any(d < 1e-12):
            break
        J = np.column_stack([-(r - rc) / d, -(z - zc) / d, -np.ones_like(d)])
        try:
            upd, *_ = np.linalg.lstsq(J, -(d - R), rcond=None)
        except np.linalg.LinAlgError:
            break
        rc += upd[0]; zc += upd[1]; R += upd[2]
        if np.linalg.norm(upd) < 1e-10:
            break
    if not (np.isfinite(rc) and np.isfinite(zc) and np.isfinite(R)) or R <= 0:
        return None
    rms = float(np.sqrt(np.mean((np.hypot(r - rc, z - zc) - R) ** 2)))
    return float(rc), float(zc), float(R), rms


def circle_contact(rc, zc, R, side, near_r=None):
    """Contact point and angle where the circle meets z = 0."""
    if abs(zc) >= R:
        return None, None
    off = math.sqrt(R * R - zc * zc)
    cands = (rc - off, rc + off)
    if near_r is not None:
        r0 = min(cands, key=lambda c: abs(c - near_r))
    else:
        r0 = cands[0] if side == 'left' else cands[1]
    # tangent is perpendicular to the radius (r0 - rc, -zc)
    tr, tz = zc, (r0 - rc)
    return angle_from_tangent(tr, tz, side), float(r0)


# -------------------------------------------------------------------- ellipse
def fit_ellipse(r: np.ndarray, z: np.ndarray):
    """Halir-Flusser direct ellipse fit. Returns conic (A, B, C, D, E, F)."""
    if r.size < 6:
        return None
    D1 = np.column_stack([r * r, r * z, z * z])
    D2 = np.column_stack([r, z, np.ones_like(r)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
    try:
        ev, evec = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    cond = 4 * evec[0] * evec[2] - evec[1] ** 2
    ok = np.where((cond > 0) & np.isfinite(ev))[0]
    if ok.size == 0:
        return None
    a1 = np.real(evec[:, ok[0]])
    co = np.concatenate([a1, T @ a1])
    return co if np.all(np.isfinite(co)) else None


def conic_rms(co, r, z) -> float:
    A, B, C, D, E, F = co
    v = A * r * r + B * r * z + C * z * z + D * r + E * z + F
    # normalise by the gradient so the residual is roughly a distance
    gr = 2 * A * r + B * z + D
    gz = B * r + 2 * C * z + E
    g = np.hypot(gr, gz) + 1e-12
    return float(np.sqrt(np.mean((v / g) ** 2)))


def conic_contact(co, side, near_r=None):
    """Where the conic crosses z = 0, and the tangent angle there."""
    A, B, C, D, E, F = co
    if abs(A) < 1e-15:
        return None, None
    disc = D * D - 4 * A * F
    if disc < 0:
        return None, None
    s = math.sqrt(disc)
    cands = ((-D - s) / (2 * A), (-D + s) / (2 * A))
    if near_r is not None:
        r0 = min(cands, key=lambda c: abs(c - near_r))
    else:
        r0 = min(cands) if side == 'left' else max(cands)
    Fr = 2 * A * r0 + D          # dF/dr at (r0, 0)
    Fz = B * r0 + E              # dF/dz at (r0, 0)
    # tangent is perpendicular to the conic gradient
    return angle_from_tangent(-Fz, Fr, side), float(r0)


# ----------------------------------------------------------------- polynomial
def fit_poly(r: np.ndarray, z: np.ndarray, deg: int = 2):
    """r = f(z). Returns (coeffs, r0, dr/dz at z=0, rms)."""
    if r.size < deg + 2:
        return None
    try:
        co = np.polyfit(z, r, deg)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(co)):
        return None
    d = np.polyder(co)
    rms = float(np.sqrt(np.mean((np.polyval(co, z) - r) ** 2)))
    return co, float(np.polyval(co, 0.0)), float(np.polyval(d, 0.0)), rms


def poly_contact(co, side):
    d = np.polyder(co)
    slope = float(np.polyval(d, 0.0))
    return angle_from_tangent(slope, 1.0, side), float(np.polyval(co, 0.0))


# ----------------------------------------------------------------------- line
def fit_line(r: np.ndarray, z: np.ndarray):
    """r = m*z + c, total-least-squares free. Returns (m, c, max_abs_resid)."""
    if r.size < 3:
        return None
    try:
        m, c = np.polyfit(z, r, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    resid = np.abs(np.polyval([m, c], z) - r)
    return float(m), float(c), float(resid.max())


def line_contact(m, c, side):
    return angle_from_tangent(m, 1.0, side), float(c)
