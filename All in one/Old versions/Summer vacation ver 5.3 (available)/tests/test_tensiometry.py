"""End-to-end tests for pendant-drop tensiometry.

The point of these tests is that they need **no experimental data**.  A
Young-Laplace shape is generated from a known Bond number, placed in an image
frame with a known apex and rotation, optionally degraded with pixel noise, and
handed to the fitter.  If the fitter cannot recover what the forward model was
told, it will not recover anything from a photograph either.

Run directly (python tests/test_tensiometry.py) or under pytest.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadrop.tensiometry import (AIR_DENSITY, GRAVITY, PendantFitResult,  # noqa: E402
                                pixel_scale_from_needle, shape_parameter,
                                shape_parameter_from_profile, surface_tension,
                                synthesise_pendant_drop, worthington_number,
                                young_laplace_fit)
from cadrop.younglaplace import YoungLaplaceShape  # noqa: E402

# a realistic pendant-drop configuration: Bo ~ 0.3 is what a water drop of a
# few millimetres gives, and 150 px per apex radius is comfortably resolved
BO_TRUE = 0.30
RADIUS_PX = 150.0
APEX = (400.0, 700.0)


def _profile(bo=BO_TRUE, radius=RADIUS_PX, apex=APEX, rotation=0.0,
             noise=0.0, seed=0, s_frac=0.9):
    sh = YoungLaplaceShape(bo)
    return synthesise_pendant_drop(bo, radius, apex, rotation_deg=rotation,
                                   s_top=s_frac * sh.s_max,
                                   n_per_branch=80, noise_px=noise, seed=seed)


def test_round_trip_recovers_parameters_without_noise():
    """With an exact profile the fitter must return what it was given.

    The tolerances here are tight on purpose: an exact synthetic profile has no
    information the fitter is not entitled to, so anything worse than a few
    parts in a thousand means the optimiser or the model sampling is losing
    precision that a real measurement would then be blamed for.
    """
    pts = _profile()
    res = young_laplace_fit(pts)

    assert res.ok, res.error
    assert res.bond is not None and abs(res.bond - BO_TRUE) / BO_TRUE < 0.005, \
        f'bond {res.bond} vs {BO_TRUE}'
    assert abs(res.radius_px - RADIUS_PX) / RADIUS_PX < 0.005, \
        f'radius {res.radius_px} vs {RADIUS_PX}'
    assert math.hypot(res.apex_x - APEX[0], res.apex_y - APEX[1]) < 1.0, \
        f'apex ({res.apex_x:.1f}, {res.apex_y:.1f}) vs {APEX}'
    assert abs(res.rotation_deg) < 0.2, f'rotation {res.rotation_deg}'
    assert res.rms_px < 0.2, f'rms {res.rms_px:.4f} px'


def test_round_trip_recovers_bond_across_the_practical_range():
    """Recovery must not be a coincidence of one Bond number."""
    for bo_true in (0.10, 0.20, 0.40, 0.50):
        pts = _profile(bo=bo_true)
        res = young_laplace_fit(pts)
        assert res.ok, f'Bo={bo_true}: {res.error}'
        err = abs(res.bond - bo_true) / bo_true
        assert err < 0.01, f'Bo={bo_true} -> {res.bond:.4f} ({err:.2%} error)'


def test_round_trip_survives_pixel_noise():
    """Half a pixel of edge noise is what a real extraction delivers."""
    errs = []
    for seed in range(4):
        pts = _profile(noise=0.5, seed=seed)
        res = young_laplace_fit(pts)
        assert res.ok, res.error
        errs.append(abs(res.bond - BO_TRUE) / BO_TRUE)
    worst = max(errs)
    assert worst < 0.15, f'worst Bo error {worst:.1%} over 4 noise seeds'


def test_round_trip_recovers_a_tilted_camera():
    """A camera that is not level must be absorbed by the rotation parameter."""
    pts = _profile(rotation=3.0)
    res = young_laplace_fit(pts)
    assert res.ok, res.error
    assert abs(res.bond - BO_TRUE) / BO_TRUE < 0.05, f'bond {res.bond}'
    assert abs(res.rotation_deg - 3.0) < 1.5, f'rotation {res.rotation_deg}'


def test_coordinate_convention_is_not_mirrored():
    """Guards the sign of z.

    The model maps z (into the liquid, i.e. upward) to a *decreasing* image y.
    If that sign is flipped the fitter happily fits an upside-down drop, so a
    vertically mirrored profile must fit noticeably worse.
    """
    pts = _profile()
    flipped = pts.copy()
    flipped[1] = 2.0 * APEX[1] - flipped[1]

    good = young_laplace_fit(pts)
    bad = young_laplace_fit(flipped)
    assert good.ok, good.error
    assert not bad.ok or bad.rms_px > 5.0 * good.rms_px, \
        f'mirrored profile fitted almost as well ({bad.rms_px:.3f} vs {good.rms_px:.3f})'


def test_fit_rejects_bad_input():
    for bad in (np.zeros((3, 20)), np.zeros((2, 4)), np.full((2, 30), np.nan)):
        res = young_laplace_fit(bad)
        assert isinstance(res, PendantFitResult) and not res.ok
        assert res.error


def test_end_to_end_surface_tension_recovery():
    """Close the loop: known gamma in, same gamma out, entirely synthetically.

    A real water/air pendant drop is simulated at a chosen gamma, imaged at a
    chosen pixel scale, fitted, and converted back to mN/m.  This is the only
    test that exercises the whole chain, so it is the one that would catch a
    units error in ``surface_tension`` or a sign error in the calibration.
    """
    gamma_true = 72.0                      # mN/m, water at ~25 C
    delta_rho = 997.0 - AIR_DENSITY
    px_per_mm = 100.0
    px_size_mm = 1.0 / px_per_mm

    for bo_true in (0.15, 0.30, 0.45):
        # forward model: pick the apex radius that produces this Bond number
        r0_m = math.sqrt(bo_true * gamma_true * 1e-3 / (delta_rho * GRAVITY))
        radius_px = r0_m * 1000.0 * px_per_mm
        pts = _profile(bo=bo_true, radius=radius_px, noise=0.4, seed=7)

        res = young_laplace_fit(pts)
        assert res.ok, res.error
        got = surface_tension(delta_rho, res.radius_px, px_size_mm, res.bond)

        err = abs(got - gamma_true) / gamma_true
        assert err < 0.05, (
            f'Bo={bo_true}: recovered gamma {got:.2f} mN/m vs {gamma_true} '
            f'({err:.2%}); fitted Bo={res.bond:.4f}, R0={res.radius_px:.1f} px')


def test_scale_error_is_doubled_into_the_answer():
    """A 1% calibration error must show up as ~2% in gamma, end to end."""
    delta_rho = 997.0 - AIR_DENSITY
    bo_true, gamma_true, px_per_mm = 0.30, 72.0, 100.0
    r0_m = math.sqrt(bo_true * gamma_true * 1e-3 / (delta_rho * GRAVITY))
    radius_px = r0_m * 1000.0 * px_per_mm

    pts = _profile(bo=bo_true, radius=radius_px)
    res = young_laplace_fit(pts)
    assert res.ok, res.error

    good = surface_tension(delta_rho, res.radius_px, 1.0 / px_per_mm, res.bond)
    # pretend the needle was 1% wider than assumed
    bad = surface_tension(delta_rho, res.radius_px, 1.01 / px_per_mm, res.bond)

    ratio = bad / good
    assert abs(ratio - 1.0201) < 1e-6, ratio


def test_shape_parameter_from_profile_matches_the_exact_shape():
    """The data-driven P_s must agree with the closed-form one.

    ``shape_parameter_from_profile`` exists so the conditioning question can be
    answered on images that already exist, without fitting anything.  It uses a
    polygon area and a circle fit instead of the exact quadrature, so it is
    approximate — but it must be close enough that a threshold decision is not
    flipped.  Tolerance is 15%; observed agreement is ~7%.
    """
    radius = 150.0
    for bo in (0.05, 0.10, 0.20, 0.30, 0.50):
        sh = YoungLaplaceShape(bo)
        s = np.linspace(0.0, sh.s_max, 140)
        r, z, _ = sh.profile(s)
        zz = z.max() - z                       # apex at the top, sessile layout
        prof = np.concatenate([np.stack([radius * r, radius * zz]),
                               np.stack([-radius * r, radius * zz])], axis=1)

        exact = shape_parameter(sh, sh.s_max)
        got, r0 = shape_parameter_from_profile(prof)

        assert got is not None and r0 is not None, f'Bo={bo} returned nothing'
        assert abs(r0 - radius) / radius < 0.05, f'Bo={bo}: R0 {r0:.1f} vs {radius}'
        assert abs(got - exact) / exact < 0.15, \
            f'Bo={bo}: P_s {got:.4f} vs exact {exact:.4f}'


def test_shape_parameter_from_profile_rejects_junk():
    for bad in (np.zeros((3, 40)), np.zeros((2, 5)), np.full((2, 40), np.nan)):
        assert shape_parameter_from_profile(bad) == (None, None)


# --------------------------------------------------------- derived quantities
def test_shape_parameter_zero_for_a_full_sphere():
    """P_s is defined to vanish for a sphere, which is the Bo = 0, s = pi case."""
    sh = YoungLaplaceShape(0.0)
    assert abs(sh.s_max - math.pi) < 1e-6
    assert abs(shape_parameter(sh, sh.s_max)) < 1e-6


def test_shape_parameter_grows_with_deformation():
    """More gravity deformation must mean more available shape information."""
    vals = []
    for bo in (0.05, 0.10, 0.20, 0.30):
        sh = YoungLaplaceShape(bo)
        vals.append(shape_parameter(sh, sh.s_max))
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    assert vals[0] < 0.10 and vals[-1] > 0.25, vals


def test_surface_tension_is_quadratic_in_the_scale():
    """gamma ~ scale**2, so a relative scale error is doubled in gamma.

    This is the single most important sensitivity in the whole measurement, and
    it is cheap to pin down here.
    """
    delta_rho = 997.0 - AIR_DENSITY
    base = surface_tension(delta_rho, 150.0, 1.0 / 100.0, 0.3)
    doubled = surface_tension(delta_rho, 150.0, 2.0 / 100.0, 0.3)
    assert abs(doubled / base - 4.0) < 1e-9, doubled / base

    # and the absolute value must be physically sensible for water at Bo=0.3
    assert 40.0 < base < 120.0, base


def test_pixel_scale_from_needle():
    # a 1.5 mm needle spanning 300 px is 0.005 mm/px
    assert abs(pixel_scale_from_needle(1.5, 300.0) - 0.005) < 1e-12
    try:
        pixel_scale_from_needle(1.5, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError('zero needle width should be rejected')


def test_worthington_number_uses_the_kratz_convention():
    """Wo = d_rho g V / (pi gamma D); check against a hand-computed value."""
    delta_rho, vol, gamma, needle = 995.8, 20e-9, 72.0, 1.5e-3
    expected = delta_rho * GRAVITY * vol / (math.pi * gamma * 1e-3 * needle)
    got = worthington_number(delta_rho, vol, gamma, needle)
    assert abs(got - expected) < 1e-12
    # 20 uL on a 1.5 mm holder is a healthy drop, not a marginal one
    assert 0.2 < got < 0.6, got


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
