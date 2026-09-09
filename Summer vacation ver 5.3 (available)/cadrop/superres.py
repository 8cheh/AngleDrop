"""Optional super-resolution preprocessing via Real-ESRGAN (ONNX + onnxruntime).

Upscales a low-resolution / blurry drop image *before* detection so the profile
edge carries more detail for the sub-pixel trace and the contact-angle fit.
This is the "first step" of preprocessing, ahead of :func:`cadrop.detect.enhance`.

The reference model is the Real-ESRGAN "general x4v3" export (community re-host,
int8-quantized, ~4.9 MB). It upscales natively by 4x; any other requested scale
is reached by resizing the 4x output.

``onnxruntime`` is an optional dependency -- import this module and call
:func:`available` to probe whether super-resolution can actually run.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - onnxruntime is optional
    ort = None

_MODEL_NAME = 'realesr-general-x4v3.onnx'
_MODEL_SCALE = 4
# Real-ESRGAN's pixel-shuffle upsampling needs input dims divisible by 4.
_PAD_ALIGN = 4


def _default_model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', 'models', _MODEL_NAME))


@lru_cache(maxsize=1)
def _session(model_path: str):
    """Load (and cache) the ONNX session. Cached so batch runs load it once."""
    if ort is None:
        return None
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


def available() -> bool:
    """Whether super-resolution can run: onnxruntime present + model on disk."""
    return ort is not None and os.path.isfile(_default_model_path())


def super_resolve(bgr: np.ndarray, scale: float = 3.0,
                  model_path: Optional[str] = None) -> np.ndarray:
    """Upscale a BGR image with Real-ESRGAN and return the upscaled BGR image.

    ``scale`` is the requested factor. The model produces 4x natively; other
    factors are obtained by resizing the 4x result.
    """
    if ort is None:
        raise RuntimeError('onnxruntime is not installed (pip install onnxruntime)')
    model_path = model_path or _default_model_path()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'super-resolution model not found: {model_path}')

    sess = _session(model_path)
    if sess is None:
        raise RuntimeError('failed to load super-resolution model')

    h, w = bgr.shape[:2]
    # Real-ESRGAN expects RGB float32 in [0, 1], NCHW.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pad_h = (-h) % _PAD_ALIGN
    pad_w = (-w) % _PAD_ALIGN
    if pad_h or pad_w:
        rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    x = rgb.transpose(2, 0, 1)[None, ...]  # 1, 3, H, W
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: x})[0]  # 1, 3, H*4, W*4

    up = out[0].transpose(1, 2, 0)           # H*4, W*4, 3 (RGB, float)
    up = np.clip(up, 0.0, 1.0)
    up = (up * 255.0 + 0.5).astype(np.uint8)
    up = cv2.cvtColor(up, cv2.COLOR_RGB2BGR)

    # crop the reflect-padding back off (scaled by the same factor)
    up_h, up_w = up.shape[:2]
    crop_h = pad_h * _MODEL_SCALE
    crop_w = pad_w * _MODEL_SCALE
    if crop_h or crop_w:
        up = up[0:up_h - crop_h, 0:up_w - crop_w]

    if scale != _MODEL_SCALE:
        th = max(1, int(round(h * scale)))
        tw = max(1, int(round(w * scale)))
        up = cv2.resize(up, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    return up
