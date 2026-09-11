"""预处理核心函数：超分辨率 + 液滴定位。

对照仓库：
  * `read_image`        —— 与 `_src/cadrop_detect.py::read_image` 一致
  * `super_resolve`     —— 与 GitHub 仓库 `cadrop/superres.py` 一致（本地 `_src` 没有）
                            FSRCNN 只对液滴 ROI 超分，其余快速插值；
                            无 dnn_superres / 模型时回退到经典插值+细节增强
  * `locate_drop`       —— 移植 `_src/cadrop_detect.py` 的「基线检测 + 液滴分割」
                            联合搜索（`find_substrate` + `_segment`），输出液滴包围盒
  * `process`           —— 把上面两步串成一条预处理流水线，供 UI 展示

只做预处理：超分辨（可选）+ 定位，不做灰度/Sobel/Otsu/拟合等后续步骤。
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

__all__ = ['read_image', 'sr_available', 'super_resolve', 'locate_drop', 'process']


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def read_image(path: str) -> Optional[np.ndarray]:
    """读取 BGR 图像，容忍 Windows 下非 ASCII 路径。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
# 超分辨率（GitHub 仓库独有，本地 _src 没有）
# --------------------------------------------------------------------------- #
_MODEL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models'))


def _model_path(scale: int) -> str:
    return os.path.join(_MODEL_DIR, f'FSRCNN_x{scale}.pb')


def sr_available() -> bool:
    """FSRCNN 是否可用（需 opencv-contrib 的 dnn_superres + 模型文件）。"""
    return hasattr(cv2, 'dnn_superres') and os.path.isfile(_model_path(3))


@lru_cache(maxsize=3)
def _sr(scale: int):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    src = _model_path(scale)
    try:
        src.encode('ascii')
    except UnicodeEncodeError:
        tmp = os.path.join(os.environ.get('TEMP', os.path.expanduser('~')), 'fsrcnn_models')
        os.makedirs(tmp, exist_ok=True)
        dst = os.path.join(tmp, f'FSRCNN_x{scale}.pb')
        if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
            import shutil
            shutil.copyfile(src, dst)
        src = dst
    sr.readModel(src)
    sr.setModel('fsrcnn', scale)
    return sr


def _classic_upscale(bgr: np.ndarray, scale: int) -> np.ndarray:
    """无 dnn_superres 时的超分回退：三次插值 + 细节增强 + 反锐化。"""
    h, w = bgr.shape[:2]
    up = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    up = cv2.detailEnhance(up, sigma_s=8, sigma_r=0.08)
    blur = cv2.GaussianBlur(up, (0, 0), 3)
    up = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    return np.clip(up, 0, 255).astype(np.uint8)


def super_resolve(bgr: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """按 `scale` 放大整图：液滴 ROI 用 FSRCNN，其余快速插值（无则整体回退）。"""
    scale = int(round(scale))
    if scale <= 1:
        return bgr
    if not sr_available():
        return _classic_upscale(bgr, scale)

    roi = _drop_bbox(bgr)
    if roi is None:
        return _sr(scale).upsample(bgr)

    x0, y0, x1, y1 = roi
    h, w = bgr.shape[:2]
    full = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    crop = bgr[y0:y1, x0:x1]
    up = _sr(scale).upsample(crop)
    full[y0 * scale:y0 * scale + up.shape[0],
         x0 * scale:x0 * scale + up.shape[1]] = up
    return full


def _drop_bbox(bgr: np.ndarray, pad: int = 20) -> Optional[Tuple[int, int, int, int]]:
    """先定位液滴，返回带 padding 的包围盒（供 FSRCNN 只对 ROI 超分）。"""
    loc = locate_drop(bgr)
    if not loc['ok']:
        return None
    x0, y0, x1, y1 = loc['bbox']
    h, w = bgr.shape[:2]
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)
    if x1 - x0 < 30 or y1 - y0 < 30:
        return None
    return x0, y0, x1, y1


# --------------------------------------------------------------------------- #
# 液滴定位（移植自 _src/cadrop_detect.py 的 find_substrate + _segment）
# --------------------------------------------------------------------------- #
def _candidate_rows(gray: np.ndarray, max_cands: int = 8) -> np.ndarray:
    """找最强的水平「暗→亮」台阶行，按强度降序、彼此间隔 >30 行。"""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    sob = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    prof = sob.sum(axis=1)
    prof = np.convolve(prof, np.ones(15) / 15.0, mode='same')
    order = np.argsort(prof)[::-1]
    rows = []
    for r in order:
        if not rows or all(abs(r - q) > 30 for q in rows):
            rows.append(int(r))
        if len(rows) >= max_cands:
            break
    return np.array(rows, dtype=int)


def _refine_line(gray: np.ndarray, row: float, polarity: int, seed: int,
                 win: int = 40, max_slope: float = 0.12,
                 sob: Optional[np.ndarray] = None) -> Optional[Tuple[float, float]]:
    """在 `row` 附近用 RANSAC 拟合近水平台面边缘，返回 (k, b)。

    polarity=+1：暗上亮下（正立液滴）；-1 相反。
    """
    h, w = gray.shape
    if sob is None:
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        sob = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    sob = sob * polarity

    y0 = int(max(0, row - win))
    y1 = int(min(h, row + win))
    if y1 - y0 < 10:
        return None
    band = sob[y0:y1, :]
    xs = np.arange(0, w, 3)
    sub = band[:, xs]
    cols = np.arange(sub.shape[1])
    idx = np.argmax(sub, axis=0)
    peak = sub[idx, cols]

    a = sub[np.clip(idx - 1, 0, None), cols]
    c = sub[np.clip(idx + 1, None, sub.shape[0] - 1), cols]
    den = a - 2 * peak + c
    off = np.where(np.abs(den) > 1e-6,
                   0.5 * (a - c) / np.where(np.abs(den) > 1e-6, den, 1.0), 0.0)
    ys = y0 + idx + np.clip(off, -1, 1)

    strong = peak > max(8.0, float(np.percentile(peak, 50)))
    X, Y = xs[strong].astype(float), ys[strong]
    if X.size < 15:
        X, Y = xs.astype(float), ys
    if X.size < 4:
        return None

    rng = np.random.default_rng(seed)
    best_in, best = None, None
    n = X.size
    for _ in range(200):
        i, j = rng.integers(0, n, 2)
        if abs(X[i] - X[j]) < w * 0.2:
            continue
        k = (Y[j] - Y[i]) / (X[j] - X[i])
        if abs(k) > max_slope:
            continue
        b = Y[i] - k * X[i]
        inl = np.abs(Y - (k * X + b)) < 3.0
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best = inl, (float(k), float(b))
    if best_in is None or best_in.sum() < 10:
        return None
    k, b = np.polyfit(X[best_in], Y[best_in], 1)
    if abs(k) > max_slope:
        return None
    return float(k), float(b)


def _z_map(k: float, b: float, w: int, h: int, invert: bool) -> np.ndarray:
    """每像素到基线的垂向距离；>0 表示「液滴一侧」（正立为线上方）。"""
    xx = np.arange(w, dtype=float)[None, :]
    yy = np.arange(h, dtype=float)[:, None]
    dx = w - 1.0
    dy = k * (w - 1.0)
    n = math.hypot(dx, dy)
    if n < 1e-9:
        n = 1.0
    ux, uy = dx / n, dy / n
    px, py = uy, -ux
    if invert:
        px, py = -px, -py
    return px * xx + py * (yy - b)


def _segment(gray: np.ndarray, k: float, b: float, invert: bool,
             clearance: float = 3.0, min_area: int = 400):
    """在基线「液滴一侧」做 Otsu + 连通域约束，返回 (mask, score)。"""
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    zmap = _z_map(k, b, w, h, invert)
    inside = zmap > clearance

    vals = g[inside]
    if vals.size < 500:
        return None, -1.0
    thr, _ = cv2.threshold(vals.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best = None
    for polarity in (1, -1):
        m = ((g >= thr) if polarity > 0 else (g <= thr)) & inside
        m = (m * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x0, y0, bw, bh, area = stats[i]
            if area < min_area or bh < 8:
                continue
            if bw > 0.75 * w:
                continue
            if x0 <= 1 or x0 + bw >= w - 1 or y0 <= 1:
                continue
            cy = y0 + bh / 2.0
            col = x0 + bw / 2.0
            if zmap[int(np.clip(cy, 0, h - 1)), int(np.clip(col, 0, w - 1))] < clearance:
                continue
            yy0, yy1 = max(0, y0 - 2), min(h, y0 + bh + 2)
            xx0, xx1 = max(0, x0), min(w, x0 + bw)
            if zmap[yy0:yy1, xx0:xx1][(lab[yy0:yy1, xx0:xx1] == i)].min() > clearance + 4.0:
                continue
            ar = bw / max(bh, 1)
            score = float(area) / (1.0 + max(0.0, ar - 2.0))
            score *= 1.0 - min(1.0, max(0.0, (bw - 0.7 * w) / (0.3 * w)))
            if best is None or score > best[0]:
                best = (score, (lab == i).astype(np.uint8) * 255)
    if best is None:
        return None, -1.0
    return best[1], best[0]


def locate_drop(bgr: np.ndarray, *, max_cands: int = 8, seed: int = 0) -> Dict:
    """定位液滴：联合搜索「台面基线 + 液滴分割」，返回包围盒等信息。

    返回 dict：
        ok, bbox=[x0,y0,x1,y1], mask(0/255), k, b, invert, score, area,
        error（失败时）
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    sob = cv2.Sobel(cv2.GaussianBlur(g, (5, 5), 0), cv2.CV_32F, 0, 1, ksize=5)

    best = None
    for row in _candidate_rows(g, max_cands):
        for polarity in (+1, -1):
            fit = _refine_line(g, row, polarity, seed, sob=sob)
            if fit is None:
                continue
            k, b = fit
            mask, score = _segment(g, k, b, invert=(polarity < 0))
            if mask is None:
                continue
            if best is None or score > best[0]:
                best = (score, k, b, polarity < 0, mask)

    # 兜底：只用正立取向
    if best is None:
        for row in _candidate_rows(g, max_cands):
            fit = _refine_line(g, row, +1, seed)
            if fit is None:
                continue
            k, b = fit
            mask, score = _segment(g, k, b, invert=False)
            if mask is not None:
                best = (score, k, b, False, mask)
                break

    if best is None:
        return {'ok': False, 'error': '未找到液滴'}

    score, k, b, invert, mask = best
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return {'ok': False, 'error': '掩码为空'}

    return {
        'ok': True,
        'bbox': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        'mask': mask,
        'k': float(k), 'b': float(b), 'invert': bool(invert),
        'score': float(score), 'area': int(mask.sum() // 255),
    }


# --------------------------------------------------------------------------- #
# 注释 / 可视化
# --------------------------------------------------------------------------- #
def annotate(bgr: np.ndarray, loc: Dict) -> np.ndarray:
    """在图上画出液滴包围盒 + 基线 + 掩码，返回可视化图。"""
    vis = bgr.copy()
    h, w = vis.shape[:2]
    if not loc.get('ok'):
        cv2.putText(vis, 'FAILED: ' + str(loc.get('error', 'no drop')),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
        return vis

    # 掩码半透明高亮
    mask = loc['mask']
    if mask.shape[:2] == (h, w):
        vis[mask > 0] = (vis[mask > 0] * 0.55 + np.array([80, 220, 80]) * 0.45).astype(np.uint8)

    # 基线
    k, b = loc['k'], loc['b']
    cv2.line(vis, (0, int(round(b))), (w - 1, int(round(k * (w - 1) + b))),
             (80, 80, 255), 3, cv2.LINE_AA)

    # 包围盒
    x0, y0, x1, y1 = loc['bbox']
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 3, cv2.LINE_AA)

    s = max(1.0, w / 1200.0)
    txt = [f'drop bbox: ({x0},{y0}) - ({x1},{y1})',
           f'size: {x1 - x0} x {y1 - y0} px   area: {loc["area"]} px2',
           f'baseline: y = {k:+.4f} x + {b:.1f}']
    yy = int(36 * s)
    for t in txt:
        cv2.putText(vis, t, (int(16 * s), yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * s, (245, 245, 245), max(1, int(1.6 * s)), cv2.LINE_AA)
        yy += int(30 * s)
    return vis


# --------------------------------------------------------------------------- #
# 预处理流水线：超分 + 定位
# --------------------------------------------------------------------------- #
def process(bgr: np.ndarray, *, use_sr: bool = False, sr_scale: float = 3.0) -> Dict:
    """预处理流水线（只做超分 + 定位）。

    返回:
        {
          'stages': [(name, bgr_image, desc), ...],   # 原图 / 超分 / 定位结果
          'loc': locate_drop 的结果,
          'sr_active': 是否走了 FSRCNN（False 表示回退或未启用）,
          'ok': bool,
        }
    """
    stages: List[Tuple[str, np.ndarray, str]] = []
    h, w = bgr.shape[:2]
    stages.append(('原图', bgr.copy(), f'{w}×{h}'))

    work = bgr
    sr_ok = False
    if use_sr and sr_scale > 1:
        sr_ok = sr_available()
        work = super_resolve(work, sr_scale)
        tag = '超分辨率' if sr_ok else '超分辨率（经典回退）'
        stages.append((tag, work.copy(),
                       f'{work.shape[1]}×{work.shape[0]}，scale={int(round(sr_scale))}'))

    loc = locate_drop(work)
    vis = annotate(work, loc)
    desc = ('定位成功 ' + str(loc['bbox'])) if loc['ok'] else '定位失败'
    stages.append(('液滴定位', vis, desc))

    return {'stages': stages, 'loc': loc, 'sr_active': sr_ok, 'ok': loc['ok']}
