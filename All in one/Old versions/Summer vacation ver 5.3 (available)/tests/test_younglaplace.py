"""Correctness tests for the Young-Laplace solver.

The sharpest test available is the ``Bo = 0`` identity: with no gravity the
Young-Laplace interface must be *exactly* the unit sphere
``r = sin(s), z = 1 - cos(s), phi = s``.  That is a closed form, not a
numerical reference, so it pins the ODE, the apex series, the sign convention
and the integration tolerances all at once.

Run directly (python tests/test_younglaplace.py) or under pytest.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadrop.younglaplace import (PENDANT, SESSILE, YoungLaplaceShape,  # noqa: E402
                                 sphere_profile)


def test_zero_bond_is_exactly_a_unit_sphere():
    """Bo = 0 has a closed-form solution; we must reproduce it to ~1e-8."""
    sh = YoungLaplaceShape(0.0, invert=True)
    s = np.linspace(0.0, 0.5 * math.pi, 200)
    r, z, phi = sh.profile(s)
    r_ex, z_ex, p_ex = sphere_profile(s)

    err_r = float(np.abs(r - r_ex).max())
    err_z = float(np.abs(z - z_ex).max())
    err_p = float(np.abs(phi - p_ex).max())

    assert err_r < 1e-8, f'r error {err_r:.3e}'
    assert err_z < 1e-8, f'z error {err_z:.3e}'
    assert err_p < 1e-8, f'phi error {err_p:.3e}'


def test_apex_curvature_limit():
    """L'Hopital gives (dphi/ds)(0) = 1; the apex must not be singular."""
    sh = YoungLaplaceShape(0.0)
    eps = 1e-4
    r, z, phi = sh.profile([eps])
    assert abs(phi[0] / eps - 1.0) < 1e-3, f'dphi/ds(0) ~ {phi[0] / eps}'
    assert abs(r[0] / eps - 1.0) < 1e-3, f'dr/ds(0) ~ {r[0] / eps}'
    assert abs(z[0] / (0.5 * eps * eps) - 1.0) < 1e-2, f'z ~ eps^2/2 failed'


def test_sphere_volume_and_area_at_equator():
    """Half the unit sphere: V = 2*pi/3, A = 2*pi."""
    sh = YoungLaplaceShape(0.0)
    eq = sh.equator()
    assert eq is not None and abs(eq - 0.5 * math.pi) < 1e-6, f'equator at {eq}'
    assert abs(sh.volume(eq) - 2.0 * math.pi / 3.0) < 1e-6, sh.volume(eq)
    assert abs(sh.surface_area(eq) - 2.0 * math.pi) < 1e-6, sh.surface_area(eq)


def test_pendant_elongates_with_bond_number():
    """Gravity stretches a hanging drop: the equator moves away from the apex.

    This is the shape information that carries the surface tension, so it is
    worth asserting directly rather than only through the fitter.
    """
    z_eq = []
    for bo in (0.0, 0.05, 0.1, 0.2, 0.5):
        sh = YoungLaplaceShape(bo, invert=True)
        eq = sh.equator()
        assert eq is not None, f'no equator found at Bo={bo}'
        z_eq.append(float(sh.profile([eq])[1][0]))

    assert all(b > a for a, b in zip(z_eq, z_eq[1:])), f'not monotone: {z_eq}'
    # Bo = 0 gives the sphere, whose equator sits exactly one radius up
    assert abs(z_eq[0] - 1.0) < 1e-6, z_eq[0]


def test_sessile_and_pendant_differ_in_the_right_direction():
    """Both signs must be implemented, and they must not be interchangeable."""
    shared = dict(s_max=6.0)
    z_pendant = YoungLaplaceShape(0.3, invert=True, **shared)
    z_sessile = YoungLaplaceShape(0.3, invert=False, **shared)

    eq_p, eq_s = z_pendant.equator(), z_sessile.equator()
    assert eq_p is not None and eq_s is not None

    # a pendant drop hangs longer than the sphere; a sessile drop is flattened
    z_p = float(z_pendant.profile([eq_p])[1][0])
    z_s = float(z_sessile.profile([eq_s])[1][0])
    assert z_p > 1.0 > z_s, f'pendant z={z_p:.3f}, sessile z={z_s:.3f}'

    assert z_pendant.sigma == PENDANT and z_sessile.sigma == SESSILE


def test_bond_sensitivity_matches_finite_differences():
    """The integrated dr/dBo, dz/dBo must equal a numerical derivative."""
    bo, h = 0.4, 1e-5
    s = np.linspace(0.0, 2.0, 25)

    a = YoungLaplaceShape(bo - h)
    b = YoungLaplaceShape(bo + h)
    r_a, z_a, _ = a.profile(s)
    r_b, z_b, _ = b.profile(s)

    m = YoungLaplaceShape(bo)
    dr, dz = m.profile_dBo(s)

    fd_r = (r_b - r_a) / (2 * h)
    fd_z = (z_b - z_a) / (2 * h)

    # skip the first few points, where both profiles are ~0 and the relative
    # comparison is dominated by round-off
    assert np.allclose(dr[5:], fd_r[5:], rtol=1e-4, atol=1e-6), \
        f'max dr diff {np.abs(dr[5:] - fd_r[5:]).max():.3e}'
    assert np.allclose(dz[5:], fd_z[5:], rtol=1e-4, atol=1e-6), \
        f'max dz diff {np.abs(dz[5:] - fd_z[5:]).max():.3e}'


def test_closest_recovers_the_generating_arc_length():
    """A point taken off the meridian must project back onto its own s."""
    sh = YoungLaplaceShape(0.25)
    for s_true in (0.3, 0.8, 1.4, 2.0):
        r, z, _ = sh.profile([s_true])
        s_got, dist = sh.closest(float(r[0]), float(z[0]))
        assert dist < 1e-6, f'distance {dist:.3e} at s={s_true}'
        assert abs(s_got - s_true) < 1e-3, f's {s_got:.5f} vs {s_true}'


def test_closest_is_sane_for_a_point_off_the_curve():
    """An off-curve point must return a positive distance, not blow up."""
    sh = YoungLaplaceShape(0.3)
    s, d = sh.closest(5.0, 5.0)
    assert math.isfinite(s) and math.isfinite(d) and d > 0
    assert 0.0 <= s <= sh.s_max


def test_rejects_negative_bond():
    for bad in (-1.0, float('nan'), float('inf')):
        try:
            YoungLaplaceShape(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad!r} should have been rejected')


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
        except AssertionError as e:
            bad += 1
            print(f'  FAIL  {fn.__name__}: {e}')
    print(f'{len(fns) - bad}/{len(fns)} passed')
    sys.exit(1 if bad else 0)
