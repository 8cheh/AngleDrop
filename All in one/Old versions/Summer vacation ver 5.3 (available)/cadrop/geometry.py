"""Baseline geometry and the image <-> baseline coordinate transform.

Baseline coordinates (r, z):
    r  runs along the substrate, increasing to the right
    z  runs perpendicular to it, increasing UPWARD (i.e. towards smaller image y)

The substrate is z = 0, so a contact angle is simply the angle between the
fitted profile tangent at z = 0 and the r axis, measured through the liquid.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ['Line']


class Line:
    """A straight substrate line, y = k*x + b in image coordinates."""

    def __init__(self, k: float, b: float, width: float, invert: bool = False):
        self.k = float(k)
        self.b = float(b)
        self.invert = bool(invert)
        p0 = np.array([0.0, float(b)])
        p1 = np.array([float(width - 1), k * (width - 1) + b])
        d = p1 - p0
        n = math.hypot(*d)
        if n < 1e-9:
            raise ValueError('degenerate baseline')
        self.origin = p0
        self.unit = d / n
        # perp = (uy, -ux) points towards smaller y, i.e. "up" on screen.
        # For an inverted drop (hanging below the substrate) it points down, so
        # that z > 0 always means "inside the liquid" and every downstream
        # formula is orientation-agnostic. r still grows to the right on screen,
        # so left/right contact semantics are unchanged.
        self.perp = np.array([self.unit[1], -self.unit[0]])
        if invert:
            self.perp = -self.perp

    @property
    def angle_deg(self) -> float:
        """Substrate tilt in degrees (positive = rises to the right)."""
        return -math.degrees(math.atan2(self.unit[1], self.unit[0]))

    def y_at(self, x):
        return self.k * np.asarray(x, dtype=float) + self.b

    def to_rz(self, xy: np.ndarray) -> np.ndarray:
        """(2, N) image points -> (2, N) baseline coordinates."""
        xy = np.asarray(xy, dtype=float)
        rel = xy - self.origin[:, None]
        return np.stack([self.unit @ rel, self.perp @ rel])

    def to_xy(self, rz) -> np.ndarray:
        """(2, N) or (2,) baseline coordinates -> image points."""
        rz = np.asarray(rz, dtype=float)
        single = rz.ndim == 1
        if single:
            rz = rz[:, None]
        out = (self.origin[:, None]
               + self.unit[:, None] * rz[0]
               + self.perp[:, None] * rz[1])
        return out[:, 0] if single else out

    def baseline_y_map(self, shape) -> np.ndarray:
        """(h, w) array giving the substrate y for every pixel column."""
        h, w = shape
        xx = np.arange(w, dtype=float)[None, :]
        return np.repeat(self.y_at(xx), h, axis=0)

    def z_map(self, shape) -> np.ndarray:
        """(h, w) perpendicular distance from the substrate; > 0 is inside the drop."""
        h, w = shape
        xx = np.arange(w, dtype=float)[None, :]
        yy = np.arange(h, dtype=float)[:, None]
        return self.perp[0] * (xx - self.origin[0]) + self.perp[1] * (yy - self.origin[1])
