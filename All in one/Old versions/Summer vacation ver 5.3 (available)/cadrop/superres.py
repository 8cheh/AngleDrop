"""Lightweight super-resolution preprocessing via FSRCNN (OpenCV dnn_superres).

FSRCNN is a small, fast SR network: ~41KB model, roughly 10x faster than
Real-ESRGAN, landing in the tens-to-hundreds-of-ms range. Reconstruction
quality is lower than Real-ESRGAN -- that is the trade-off for speed.

Super-resolution runs on the *drop ROI only*: the drop is located first, then
only that patch is fed to the network (the rest of the frame is upscaled by
fast interpolation), which is what keeps the total time in the ms range.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

_MODEL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
_ASCII_DIR = os.path.join(os.environ.get('TEMP', os.path.expanduser('~')), 'fsrcnn_models')


def _model_path(scale: int) -> str:
    return os.path.join(_MODEL_DIR, f'FSRCNN_x{scale}.pb')


def _readable_model_path(scale: int) -> str:
    """Return a path cv2 can open.

    cv2 fails to open absolute paths containing non-ASCII characters on
    Windows (e.g. a project folder with a Chinese name). If the model path is
    non-ASCII, copy it into an ASCII temp dir and load from there.
    """
    src = _model_path(scale)
    try:
        src.encode('ascii')
        return src
    except UnicodeEncodeError:
        pass
    os.makedirs(_ASCII_DIR, exist_ok=True)
    dst = os.path.join(_ASCII_DIR, f'FSRCNN_x{scale}.pb')
    if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
        import shutil
        shutil.copyfile(src, dst)
    return dst


@lru_cache(maxsize=3)
def _sr(scale: int):
    """Load (and cache) the FSRCNN upsampler for a given scale factor."""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(_readable_model_path(scale))
    sr.setModel('fsrcnn', scale)
    return sr


def available() -> bool:
    """Whether FSRCNN super-resolution can run (dnn_superres + model present)."""
    return hasattr(cv2, 'dnn_superres') and os.path.isfile(_model_path(3))


def _drop_bbox(bgr: np.ndarray, pad: int = 20) -> Optional[Tuple[int, int, int, int]]:
    """Locate the drop and return its padded bbox (x0, y0, x1, y1), or None."""
    from .detect import find_substrate
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found = find_substrate(gray)
    if found is None:
        return None
    _, mask, _, _ = found
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    h, w = bgr.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)
    if x1 - x0 < 30 or y1 - y0 < 30:
        return None
    return x0, y0, x1, y1


def super_resolve(bgr: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Upscale the image by `scale`, running FSRCNN on the drop ROI only.

    Falls back to full-frame FSRCNN if the drop can't be located.
    """
    scale = int(round(scale))
    roi = _drop_bbox(bgr)
    if roi is None:
        return _sr(scale).upsample(bgr)

    x0, y0, x1, y1 = roi
    h, w = bgr.shape[:2]
    # fast-interpolate the full frame, then paste the sharp upscaled drop patch
    full = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    crop = bgr[y0:y1, x0:x1]
    up = _sr(scale).upsample(crop)
    full[y0 * scale:y0 * scale + up.shape[0],
         x0 * scale:x0 * scale + up.shape[1]] = up
    return full


def upscale_roi(bgr: np.ndarray, mask: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Upscale using a *given* drop mask (no re-localisation).

    The full frame is upscaled by fast interpolation; the drop patch, located
    from `mask`, is upscaled by FSRCNN and pasted back. Used by the measure
    pipeline so `find_substrate` runs only once and its mask is reused.
    """
    scale = int(round(scale))
    h, w = bgr.shape[:2]
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    pad = 20
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)
    crop = bgr[y0:y1, x0:x1]
    up = _sr(scale).upsample(crop)
    full = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    full[y0 * scale:y0 * scale + up.shape[0],
         x0 * scale:x0 * scale + up.shape[1]] = up
    return full
