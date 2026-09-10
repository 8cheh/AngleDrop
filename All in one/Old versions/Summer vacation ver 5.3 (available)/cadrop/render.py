"""Annotated overlay for a measurement result."""
from __future__ import annotations

import cv2
import numpy as np

from .measure import MeasureResult

__all__ = ['annotate']

COLORS = {
    'baseline': (80, 80, 255),      # BGR
    'profile': (120, 255, 120),
    'curve': (0, 220, 255),
    'tangent': (255, 170, 40),
    'contact': (255, 90, 220),
    'text': (245, 245, 245),
}


def _poly(img, pts, color, lw):
    if pts is None or pts.shape[1] < 2:
        return
    a = np.ascontiguousarray(pts.T.reshape(-1, 1, 2).astype(np.int32))
    cv2.polylines(img, [a], False, color, lw, cv2.LINE_AA)


def annotate(bgr: np.ndarray, res: MeasureResult, *, show_profile: bool = True,
             show_curve: bool = True, show_tangent: bool = True,
             show_panel: bool = True) -> np.ndarray:
    """Draw baseline, profile, fits and a readout panel onto a copy of the image."""
    vis = bgr.copy()
    h, w = vis.shape[:2]
    s = max(1.0, w / 1200.0)
    lw = max(1, int(round(2 * s)))

    # substrate
    x0, x1 = 0, w - 1
    cv2.line(vis, (x0, int(round(res.baseline_k * x0 + res.baseline_b))),
             (x1, int(round(res.baseline_k * x1 + res.baseline_b))),
             COLORS['baseline'], lw, cv2.LINE_AA)

    if show_profile and res.profile_xy is not None:
        rad = max(1, int(round(1.6 * s)))
        for x, y in res.profile_xy.T:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(vis, (int(x), int(y)), rad, COLORS['profile'], -1, cv2.LINE_AA)

    for side in (res.left, res.right):
        if not side.ok:
            continue
        if show_curve:
            _poly(vis, side.curve_xy, COLORS['curve'], lw + 1)
        if show_tangent and side.tangent_xy is not None:
            _poly(vis, side.tangent_xy, COLORS['tangent'], lw + 1)
        if side.contact_xy is not None:
            cx, cy = int(side.contact_xy[0]), int(side.contact_xy[1])
            cv2.drawMarker(vis, (cx, cy), COLORS['contact'], cv2.MARKER_CROSS,
                           int(20 * s), lw + 1, cv2.LINE_AA)

    if not show_panel:
        return vis

    lines = []
    if res.ok:
        for tag, f in (('L', res.left), ('R', res.right)):
            if f.ok:
                lines.append(f'{tag}: {f.angle_deg:6.2f} deg  [{f.method} '
                             f'rms={f.rms_px:.2f}px n={f.n_points}]')
            else:
                lines.append(f'{tag}: fit failed')
        lines.append(f'Avg: {res.avg_angle:6.2f} deg' +
                     (f'   |L-R|={res.asymmetry:.2f}' if res.asymmetry is not None else ''))
        lines.append(f'base={res.base_width_px:.0f}px  height={res.height_px:.0f}px  '
                     f'tilt={res.baseline_tilt_deg:+.2f}deg  conf={res.confidence:.2f}')
    else:
        lines.append(f'FAILED: {res.error}')

    fs = 0.6 * s
    th = max(1, int(round(1.4 * s)))
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for t in lines]
    bw = max(x for x, _ in sizes) + int(24 * s)
    bh = int(sum(y for _, y in sizes) + len(lines) * 14 * s + 12 * s)
    pad = int(10 * s)
    ov = vis.copy()
    cv2.rectangle(ov, (pad, pad), (pad + bw, pad + bh), (18, 18, 18), -1)
    vis = cv2.addWeighted(vis, 0.38, ov, 0.62, 0)
    y = pad + int(10 * s)
    for t, (tw, tht) in zip(lines, sizes):
        y += tht + int(8 * s)
        cv2.putText(vis, t, (pad + int(12 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, COLORS['text'], th, cv2.LINE_AA)
    return vis
