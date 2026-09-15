"""Regression tests for substrate/drop detection and the confidence gate.

These pin down a real failure of the shipped example image
(`All in one/Old versions/Summer vacation ver 4.6 (available)/example picture.jpg`).
There the segmentation score was area-dominated, so a large low-contrast
background texture (58010 px) outranked the actual drop (18035 px); the
substrate line chosen was the one *below* that texture, and the frame came back
as `ok=True` with a plausible-looking but completely wrong angle.

Run directly (python tests/test_detection.py) or under pytest.
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadrop.geometry import Line  # noqa: E402


def synthetic_scene():
    """A sharp-edged convex drop plus a larger soft-edged bright region.

    Mirrors the shipped example image: the competing region is *bigger* than
    the drop, so an area-only score prefers it, but it is not a drop -- its
    boundary is a soft brightness transition rather than a real optical edge,
    and it is not convex.
    """
    h, w = 500, 800
    rng = np.random.default_rng(7)

    # soft, ragged, bright patch on the stage; reaches down across the line so
    # it satisfies the same "rests on the substrate" test the drop does
    tex = np.zeros((h, w), np.float32)
    cv2.ellipse(tex, (600, 285), (150, 100), 0, 0, 360, 1.0, -1)
    for _ in range(40):
        cx = int(rng.integers(440, 760))
        cy = int(rng.integers(180, 395))
        cv2.circle(tex, (cx, cy), int(rng.integers(10, 26)), 0.0, -1)
    tex = cv2.GaussianBlur(tex, (0, 0), 9.0)

    img = np.full((h, w), 60.0, np.float32)
    img = np.maximum(img, 60.0 + 110.0 * tex)

    # the drop: a clean, strongly-edged circle sitting on the line
    cv2.circle(img, (250, 300), 70, 220, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.uint8)
    return img, Line(0.0, 370.0, w)


def test_texture_does_not_outrank_the_drop():
    """The convex, strongly-edged drop must win over the larger ragged blob."""
    gray, line = synthetic_scene()
    mask, polarity, score = _segment_for_test(gray, line)
    assert mask is not None, 'segmentation found nothing'

    ys, xs = np.nonzero(mask > 0)
    cx = float(xs.mean())
    assert 200 < cx < 300, (
        f'segmentation picked the wrong blob: centroid x={cx:.0f}, expected ~250 '
        f'(the drop); the texture blob sits near x=600')

    # and it should actually be the drop, not a stray fragment
    assert xs.size > 8000, f'mask far too small for the drop: {xs.size} px'


def _profile(half_of_z, z_lo=3.0, z_hi=100.0):
    """Build an (r, z) profile with both left and right edge per height."""
    zs, rs = [], []
    for zz in np.arange(z_lo, z_hi):
        half = half_of_z(zz)
        zs += [zz, zz]
        rs += [400.0 - half, 400.0 + half]
    return np.stack([np.array(rs, dtype=float), np.array(zs, dtype=float)])


def test_contaminated_base_is_trimmed():
    """Rows belonging to the stage must be cut, rows belonging to the drop kept.

    Reproduces the shipped example image: the bright stage pokes a few pixels
    above the fitted line and merges with the drop, making the bottom rows far
    wider than the drop itself.
    """
    r = _profile(lambda z: 130.0 if z < 12 else 84.0)
    kept = _drop_contaminated_base(r)
    assert kept.shape[1] < r.shape[1], 'sliver was not trimmed at all'
    assert kept[1].min() >= 12.0, (
        f'trim stopped too early: lowest kept z={kept[1].min():.1f}, expected >= 12')
    assert kept[1].max() == r[1].max(), 'trim removed the top of the drop'


def test_genuine_narrowing_drop_is_not_trimmed():
    """A drop that legitimately narrows downwards must survive untouched.

    Guards the trim against firing on ordinary profiles; a hydrophilic drop is
    widest at its base, so width decreases as z increases.
    """
    r = _profile(lambda z: 85.0 - 0.10 * (z - 3.0), z_lo=3.0, z_hi=120.0)
    kept = _drop_contaminated_base(r)
    assert kept.shape[1] == r.shape[1], 'a clean drop was trimmed'
    assert kept[1].min() == r[1].min()


# --------------------------------------------------------------------- helpers
def _segment_for_test(gray, line):
    """Call cadrop.detect._segment with the score's required reference."""
    from cadrop.detect import _edge_reference, _grad_magnitude, _segment
    mag = _grad_magnitude(gray)
    return _segment(gray, line, ref_grad=_edge_reference(mag), mag=mag)


def _drop_contaminated_base(rz):
    """Resolve the trim helper wherever it lives.

    It is a measure.py helper; importing it lazily keeps this test runnable
    while the module list is in flux.
    """
    from cadrop.measure import _drop_contaminated_base as fn
    return fn(rz)


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
