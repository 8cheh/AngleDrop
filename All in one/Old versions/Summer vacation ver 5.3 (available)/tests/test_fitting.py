"""Verify contact-angle extraction against synthetic caps of known angle.

Run directly (python tests/test_fitting.py) or under pytest.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadrop import fitting as F
from cadrop.geometry import Line


def cap_points(theta_deg, R=100.0, rc=0.0, n=400, arc=1.0):
    """Points on a circular cap of contact angle theta, substrate at z = 0.

    arc < 1 keeps only the portion nearest the contact line, which is what the
    real pipeline feeds the fitter.
    """
    th = math.radians(theta_deg)
    zc = -R * math.cos(th)
    # contact points sit at phi measured from circle centre
    phi_contact = math.atan2(0 - zc, R * math.sin(th) - rc)
    phi = np.linspace(phi_contact, math.pi / 2, n) if arc >= 1 else \
        np.linspace(phi_contact, phi_contact + arc * (math.pi / 2 - phi_contact), n)
    r = rc + R * np.cos(phi)
    z = zc + R * np.sin(phi)
    return r, z


def test_circle_recovers_angle():
    for theta in (10, 25, 45, 60, 89, 90, 91, 110, 135, 160):
        r, z = cap_points(theta)
        fit = F.fit_circle(r, z)
        assert fit is not None, theta
        rc, zc, R, rms = fit
        ang, r0 = F.circle_contact(rc, zc, R, 'right', near_r=r[0])
        assert ang is not None, theta
        assert abs(ang - theta) < 0.5, f'circle right {theta} -> {ang}'
        # mirror the cap for the left side
        ang_l, _ = F.circle_contact(-rc, zc, R, 'left', near_r=-r[0])
        assert abs(ang_l - theta) < 0.5, f'circle left {theta} -> {ang_l}'


def test_obtuse_angles_are_representable():
    """The whole point of signed atan2: theta > 90 must not fold back."""
    for theta in (100, 120, 150):
        r, z = cap_points(theta)
        rc, zc, R, _ = F.fit_circle(r, z)
        ang, _ = F.circle_contact(rc, zc, R, 'right', near_r=r[0])
        assert ang > 90, f'{theta} folded to {ang}'


def test_ellipse_on_a_circle():
    for theta in (30, 60, 90, 120):
        r, z = cap_points(theta, arc=0.9)
        co = F.fit_ellipse(r, z)
        assert co is not None, theta
        ang, _ = F.conic_contact(co, 'right', near_r=r[0])
        assert ang is not None and abs(ang - theta) < 2.0, f'ellipse {theta} -> {ang}'


def test_poly_tangent_near_contact():
    for theta in (30, 60, 90, 120):
        r, z = cap_points(theta, arc=0.25)
        fit = F.fit_poly(r, z, 2)
        assert fit is not None, theta
        co, r0, slope, rms = fit
        ang, _ = F.poly_contact(co, 'right')
        assert ang is not None and abs(ang - theta) < 2.0, f'poly {theta} -> {ang}'


def test_line_fit_is_exact_for_a_wedge():
    for theta in (20, 70, 110, 150):
        z = np.linspace(0, 50, 60)
        r = 100.0 + z / math.tan(math.radians(theta)) * -1
        m, c, mx = F.fit_line(r, z)
        ang, r0 = F.line_contact(m, c, 'right')
        assert abs(ang - theta) < 1e-6, f'line {theta} -> {ang}'
        assert abs(r0 - 100.0) < 1e-6


def test_line_roundtrip_and_tilt():
    ln = Line(0.1, 50.0, 200)
    pts = np.array([[10.0, 120.0], [60.0, 30.0]])
    back = ln.to_xy(ln.to_rz(pts))
    assert np.allclose(back, pts, atol=1e-9)
    # a point above the substrate must have z > 0
    x = 100.0
    above = np.array([[x], [ln.y_at(x) - 25.0]])
    assert ln.to_rz(above)[1, 0] > 0
    assert abs(ln.angle_deg - math.degrees(math.atan(-0.1))) < 1e-9


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
