"""Young-Laplace shapes for axisymmetric drops.

The interface of a static axisymmetric drop is the solution of the
Young-Laplace equation, which balances the Laplace pressure jump across the
interface against the hydrostatic pressure difference between the two
phases. With ``s`` the arc length measured from the apex, ``phi`` the tangent
angle measured from the r axis, and all lengths scaled by the apex radius of
curvature ``R0``, the system is

    dr/ds   = cos(phi)
    dz/ds   = sin(phi)
    dphi/ds = 2 + sigma * Bo * z - sin(phi) / r

with ``r(0) = z(0) = phi(0) = 0``.  Here ``z`` is measured from the apex in
the direction pointing *into* the liquid, ``sigma = +1`` for a sessile drop
and ``sigma = -1`` for a pendant drop, and

    Bo = delta_rho * g * R0**2 / gamma

is the Bond number.

At ``Bo = 0`` gravity is absent and the solution is exactly the unit sphere
``r = sin(s)``, ``z = 1 - cos(s)``, ``phi = s``.  ``tests/test_younglaplace.py``
checks that identity: it is the sharpest available correctness test for this
module, because it is an exact closed form rather than a tolerance on a
numerical reference.

Two numerical details matter and are easy to get wrong:

* **The apex is a regular singular point.**  ``sin(phi)/r`` is 0/0 at
  ``s = 0``.  L'Hopital gives ``(dphi/ds)(0) = 1``, so integration starts at a
  small ``s = EPS`` seeded with the series ``r = eps``, ``z = eps**2/2``,
  ``phi = eps``.
* **The integration must be stopped before the profile leaves its physical
  branch.**  Past the equator the meridian narrows to a *neck* and then turns
  back outward, which is the second solution branch rather than the silhouette
  of the drop in front of the camera.  Integration therefore terminates at the
  first of: the profile closing on the axis (``r -> 0``, which is what happens
  at ``Bo = 0`` where the drop is a full sphere of arc length ``pi``), or the
  tangent falling back through ``pi/2`` (the neck).

Validated range
---------------
``Bo`` from 0 to about 0.6.  Inside that range the solver reproduces the
published selected-plane relation ``Bo = 0.1756x^2 + 0.5234x^3 - 0.2563x^4``
(``x`` = radius at height ``2*R0``, divided by ``R0``) to better than 0.4% for
every value tested, which is an independent check on the equation, the sign
convention and the apex treatment simultaneously.  ``Bo = 0`` is exact.

Above roughly 0.6 the meridian stops reaching a vertical tangent and the
profile degenerates into a semi-infinite column instead of a compact drop, so
the solver raises the arc length it reports and the shape should not be
trusted.  Typical pendant-drop tensiometry runs at ``Bo`` of 0.1-0.5 (Dekker
et al. report 0.356), so this covers the practical range; larger drops would
need the higher solution branches, which are not implemented here.

This is an independent implementation of the published mathematics
(Bashforth & Adams 1883; Rotenberg, Boruvka & Neumann, *J. Colloid Interface
Sci.* **93** (1983) 169; del Rio & Neumann, *J. Colloid Interface Sci.*
**196** (1997) 136; Berry et al., *J. Colloid Interface Sci.* **454** (2015)
226).  It contains no code from OpenDrop, which is GPL-3.0.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import simpson, solve_ivp

__all__ = ['YoungLaplaceShape', 'PENDANT', 'SESSILE']

#: Gravity adds to the apex Laplace pressure (drop sits *on* the surface).
SESSILE = +1
#: Gravity subtracts from it (drop hangs *from* a holder).
PENDANT = -1

_EPS = 1e-7          # series-expansion offset used to step off the apex
_DEFAULT_S_MAX = 6.0  # arc length cap, in units of the apex radius


class YoungLaplaceShape:
    """One Young-Laplace meridian, cached for repeated evaluation.

    Parameters
    ----------
    bond:
        Bond number ``Bo = delta_rho * g * R0**2 / gamma``.  Must be >= 0.
    invert:
        ``True`` for a pendant drop (the default), ``False`` for a sessile drop.
    s_max:
        Arc length to integrate to, in units of ``R0``.  Extended automatically
        if a later evaluation asks for more.
    """

    def __init__(self, bond: float, invert: bool = True,
                 s_max: float = _DEFAULT_S_MAX,
                 rtol: float = 1e-10, atol: float = 1e-12) -> None:
        if not np.isfinite(bond) or bond < 0.0:
            raise ValueError(f'bond must be finite and >= 0, got {bond!r}')
        self.bond = float(bond)
        self.invert = bool(invert)
        self.sigma = PENDANT if self.invert else SESSILE
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._sol = None
        self._s_max = 0.0
        self._solve(max(float(s_max), _DEFAULT_S_MAX))

    # ------------------------------------------------------------------ solve
    def _rhs(self, s: float, y: np.ndarray) -> list:
        r, z, phi, r_bo, z_bo, phi_bo = y
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        # sin(phi)/r -> 1 as s -> 0 by l'Hopital; guard the first steps
        ratio = 1.0 if r < 1e-12 else sin_phi / r
        d_phi = 2.0 + self.sigma * self.bond * z - ratio
        # sensitivities with respect to the Bond number, from differentiating
        # the system above (the reduced system OpenDrop also integrates)
        d_r_bo = -sin_phi * phi_bo
        d_z_bo = cos_phi * phi_bo
        d_phi_bo = (self.sigma * z
                    + self.sigma * self.bond * z_bo
                    - cos_phi * phi_bo / r
                    + sin_phi * r_bo / (r * r)) if r > 1e-12 else 0.0
        return [cos_phi, sin_phi, d_phi, d_r_bo, d_z_bo, d_phi_bo]

    def _solve(self, s_max: float) -> None:
        y0 = [_EPS, 0.5 * _EPS * _EPS, _EPS, 0.0, 0.0, 0.0]

        def closed(s, y):
            return y[0]                     # r returns to the axis

        closed.terminal = True
        closed.direction = -1

        def neck(s, y):
            return y[2] - 0.5 * math.pi     # phi falls back through pi/2

        neck.terminal = True
        neck.direction = -1

        sol = solve_ivp(self._rhs, (_EPS, s_max), y0, method='DOP853',
                        rtol=self._rtol, atol=self._atol,
                        events=(closed, neck), dense_output=True)
        if sol.y.shape[1] < 4:
            raise RuntimeError(f'Young-Laplace integration failed: {sol.message}')

        # The integration legitimately ends before s_max, and it must: past the
        # equator the meridian narrows to a neck and then turns back outward,
        # which is the *second* solution branch rather than the silhouette of
        # the drop in front of the camera.  Stopping at the neck is what makes
        # s_max mean "the drop", so `s_max` here is a cap, not a promise.
        self._s_max = float(sol.t[-1])
        self._sol_t, self._sol_y = sol.t, sol.y
        if sol.sol is not None:
            self._interp = sol.sol
        else:                               # pragma: no cover - solver fallback
            from scipy.interpolate import CubicSpline
            self._interp = CubicSpline(sol.t, sol.y, axis=1)
        self._refresh_samples()

    def _eval(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        out = np.array(self._interp(np.clip(s, _EPS, self._s_max)))
        # Below the series offset the dense output is undefined, so use the
        # apex expansion directly.  Its error is O(s**3), i.e. below 1e-20.
        near = s < _EPS
        if np.any(near):
            ss = s[near]
            out[0, near] = ss
            out[1, near] = 0.5 * ss * ss
            out[2, near] = ss
            out[3, near] = 0.0
            out[4, near] = 0.0
            out[5, near] = 0.0
        return out

    def _refresh_samples(self) -> None:
        n = max(256, int(64 * self._s_max))
        self._s = np.linspace(0.0, self._s_max, n)
        y = self._eval(self._s)
        self._y = y
        self.r, self.z, self.phi = y[0], y[1], y[2]
        self.dr_dBo, self.dz_dBo = y[3], y[4]

    def extend(self, s_max: float) -> None:
        """Re-integrate out to a longer arc length (a no-op if already there)."""
        if s_max > self._s_max and self._s_max >= _DEFAULT_S_MAX * 0.999:
            # only meaningful when the cap, not a closure/neck event, stopped us
            self._solve(float(s_max))

    # --------------------------------------------------------------- evaluate
    def profile(self, s) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(r, z, phi)`` at arc length(s) ``s`` (units of ``R0``)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        if s.size and s.max() > self._s_max:
            self.extend(float(s.max()) * 1.05)
        y = self._eval(s)
        return y[0], y[1], y[2]

    def profile_dBo(self, s) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(dr/dBo, dz/dBo)`` at arc length(s) ``s``."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        if s.size and s.max() > self._s_max:
            self.extend(float(s.max()) * 1.05)
        y = self._eval(s)
        return y[3], y[4]

    # ------------------------------------------------------------ derived data
    @property
    def s_max(self) -> float:
        return self._s_max

    def _quadrature(self, s: float, fn) -> float:
        """Integrate from the apex to exactly ``s`` with Simpson's rule.

        Uses its own dense grid rather than the stored samples: trapezoid on
        256 stored points leaves ~1e-4 of error on the sphere's surface area,
        which is larger than the tolerance the sphere identity deserves.
        The endpoint is included explicitly so no slice is dropped.
        """
        s = float(min(max(s, 0.0), self._s_max))
        if s <= 0.0:
            return 0.0
        n = max(513, int(2048 * s) + 1)
        ss = np.linspace(0.0, s, n)
        r, z, phi = self.profile(ss)
        return float(simpson(fn(r, z, phi), x=ss))

    def volume(self, s: float) -> float:
        """Dimensionless volume enclosed up to arc length ``s``.

        Uses ``dV/ds = pi r^2 sin(phi)`` (i.e. ``pi r^2 dz/ds``).
        """
        return self._quadrature(
            s, lambda r, z, phi: math.pi * r ** 2 * np.sin(phi))

    def surface_area(self, s: float) -> float:
        """Dimensionless surface area up to arc length ``s`` (``dA/ds = 2 pi r``)."""
        return self._quadrature(s, lambda r, z, phi: 2.0 * math.pi * r)

    def equator(self) -> Optional[float]:
        """Arc length of the widest point (``phi = pi/2``), or ``None``.

        Interpolated rather than taken from the sample grid: at 64 samples per
        unit arc length the nearest-sample estimate is off by ~6e-3, which is
        enough to fail a comparison against the exact sphere.
        """
        hit = np.nonzero(self.phi >= 0.5 * math.pi)[0]
        if not hit.size:
            return None
        i = int(hit[0])
        if i == 0:
            return 0.0
        p0, p1 = float(self.phi[i - 1]), float(self.phi[i])
        s0, s1 = float(self._s[i - 1]), float(self._s[i])
        if p1 <= p0:
            return s1
        return s0 + (0.5 * math.pi - p0) * (s1 - s0) / (p1 - p0)

    def closest(self, r: float, z: float,
                s_lo: float = 0.0) -> Tuple[float, float]:
        """Nearest point on the meridian to ``(r, z)``.

        Returns ``(s, distance)``.  Uses a dense scan followed by a local
        golden-section refinement, which is slower per call than a Newton
        iteration on the arc length but cannot diverge — and divergence here
        would silently corrupt the least-squares residuals.
        """
        if self._s.size < 2:
            return 0.0, math.inf
        # the meridian is monotone in (r, z) so a coarse scan is a good bracket
        d = np.hypot(self.r - r, self.z - z)
        i = int(np.argmin(d))

        lo = self._s[max(i - 1, 0)]
        hi = self._s[min(i + 1, self._s.size - 1)]
        if hi <= lo:
            return float(self._s[i]), float(d[i])

        # golden-section search on the squared distance
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, dd = b - inv_phi * (b - a), a + inv_phi * (b - a)
        fc = self._dist(c, r, z)
        fd = self._dist(dd, r, z)
        for _ in range(60):
            if fc < fd:
                b, dd, fd = dd, c, fc
                c = b - inv_phi * (b - a)
                fc = self._dist(c, r, z)
            else:
                a, c, fc = c, dd, fd
                dd = a + inv_phi * (b - a)
                fd = self._dist(dd, r, z)
            if b - a < 1e-12:
                break
        s = 0.5 * (a + b)
        return float(s), float(self._dist(s, r, z))

    def _dist(self, s: float, r: float, z: float) -> float:
        rs, zs, _ = self.profile([s])
        return math.hypot(rs[0] - r, zs[0] - z)


def sphere_profile(s):
    """The exact ``Bo = 0`` solution, for tests and validation."""
    s = np.asarray(s, dtype=float)
    return np.sin(s), 1.0 - np.cos(s), s
