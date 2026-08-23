#!/usr/bin/env python3
"""
core.py — 接触角测量核心算法（优化版）
=====================================

基于 OpenDrop (Berry et al., JCIS 2015; opendrop-dev/opendrop) 的算法思想
重新实现并针对透明液滴 / 深色背景 / 浅色台面图像做了优化：

优化点:
  1. 图像自适应增强   : 双边滤波去噪 + CLAHE 局部对比度增强 + 轻度锐化
  2. 鲁棒基线检测     : 行均值梯度粗定位 -> 逐列 Scharr 亚像素精定位
                        -> 迭代剔除离群点的最小二乘直线拟合
  3. 混合轮廓提取     : 首选 Otsu 亮度分割(暗背景亮液滴最稳),
                        回退 OpenDrop 风格 Scharr 梯度法
  4. 连通域智能筛选   : 底部接基线 + 高度/宽度约束锁定液滴,
                        排除台面棱线与远处反光; 相邻碎片自动合并
  5. 亚像素边缘细化   : 梯度抛物线插值, 提高接触角重复性
  6. 双模型拟合       : 圆弧拟合(几何距离 Gauss-Newton) 与二次多项式拟合并行,
                        按 RMS 残差自动择优
  7. 质量指标输出     : RMS / 置信度, 便于批量数据筛选

坐标约定:
  - 图像坐标系 (x 右, y 下)
  - 基线坐标系 (r 沿基线向右, z 垂直基线向上), 由 Line 定义
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

__all__ = [
    'Line', 'FitResult', 'MeasureResult',
    'enhance_image', 'detect_baseline', 'extract_drop_profile',
    'circle_fit', 'measure_drop', 'read_image',
]


# ===========================================================================
# 几何: 基线直线
# ===========================================================================
class Line:
    """由两点定义的直线, 并提供图像坐标 <-> 基线坐标 (r, z) 变换."""

    def __init__(self, pt0: Sequence[float], pt1: Sequence[float]):
        self.pt0 = np.asarray(pt0, dtype=float)
        self.pt1 = np.asarray(pt1, dtype=float)
        d = self.pt1 - self.pt0
        n = math.hypot(*d)
        if n < 1e-9:
            raise ValueError('基线两点重合')
        self.unit = d / n                                    # r+ 方向
        self.perp = np.array([-self.unit[1], self.unit[0]])  # 左法线
        self.origin = self.pt0.copy()

    @classmethod
    def from_slope(cls, k: float, b: float, width: float) -> 'Line':
        """由 y = k*x + b 构造."""
        return cls((0.0, b), (float(width - 1), k * (width - 1) + b))

    def to_rz(self, xy: np.ndarray) -> np.ndarray:
        """图像坐标 (2,N)/(2,) -> 基线坐标."""
        xy = np.asarray(xy, dtype=float)
        rel = xy - self.origin.reshape(2, *((1,) * (xy.ndim - 1)))
        return np.stack([np.asarray(self.unit @ rel),
                         np.asarray(self.perp @ rel)], axis=0)

    def to_xy(self, rz: np.ndarray) -> np.ndarray:
        """基线坐标 (r,z)/(2,) -> 图像坐标."""
        rz = np.asarray(rz, dtype=float)
        out = (self.origin.reshape(2, *((1,) * (rz.ndim - 1)))
               + self.unit.reshape(2, 1) * rz[0]
               + self.perp.reshape(2, 1) * rz[1])
        return out[:, 0] if rz.ndim == 1 else out


# ===========================================================================
# 数据结构
# ===========================================================================
@dataclass
class FitResult:
    """单侧拟合结果."""
    ok: bool = False
    method: str = ''                 # 'circle' / 'poly'
    angle_deg: Optional[float] = None
    contact_rz: Optional[np.ndarray] = None
    contact_xy: Optional[np.ndarray] = None
    center_xy: Optional[np.ndarray] = None
    radius_px: Optional[float] = None
    rms: float = float('inf')
    n_points: int = 0


@dataclass
class MeasureResult:
    """单张图像完整测量结果."""
    filename: str = ''
    left: FitResult = field(default_factory=FitResult)
    right: FitResult = field(default_factory=FitResult)
    avg_angle: Optional[float] = None
    baseline_k: float = 0.0
    baseline_b: float = 0.0
    n_edge_points: int = 0
    apex_xy: Optional[np.ndarray] = None
    base_width_px: Optional[float] = None
    height_px: Optional[float] = None
    confidence: float = 0.0
    error: Optional[str] = None
    annotated_bgr: Optional[np.ndarray] = None


# ===========================================================================
# 图像读取与增强
# ===========================================================================
def read_image(path: str) -> Optional[np.ndarray]:
    """支持中文路径的图像读取, 返回 BGR."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def enhance_image(img_bgr: np.ndarray) -> np.ndarray:
    """双边滤波去噪 + LAB-CLAHE + 轻度锐化, 返回增强后的 BGR 图."""
    den = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=60, sigmaSpace=60)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    out = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.5)
    out = cv2.addWeighted(out, 1.25, blur, -0.25, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


# ===========================================================================
# 基线检测
# ===========================================================================
def _refine_baseline(gray: np.ndarray, approx_y: int, win: int = 30
                     ) -> tuple[float, float]:
    """在 approx_y 附近逐列搜索梯度峰值 (含亚像素插值) 并稳健拟合直线."""
    h, w = gray.shape
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    y0, y1 = max(0, approx_y - win), min(h, approx_y + win)

    xs, ys = [], []
    for x in range(0, w, 4):
        col = sobel_y[y0:y1, x]
        if col.max() > 25:
            i = int(col.argmax())
            yi = y0 + i
            off = 0.0
            if 0 < i < col.size - 1:
                dxa, dxb, dxc = col[i - 1], col[i], col[i + 1]
                denom = dxa - 2 * dxb + dxc
                if abs(denom) > 1e-9:
                    off = float(np.clip(0.5 * (dxa - dxc) / denom, -1, 1))
            xs.append(x)
            ys.append(yi + off)

    if len(xs) < 10:
        xs = list(range(0, w, 8))
        ys = [approx_y] * len(xs)

    xs_a, ys_a = np.array(xs, float), np.array(ys, float)
    for _ in range(3):
        km, bm = np.polyfit(xs_a, ys_a, 1)
        res = np.abs(ys_a - (km * xs_a + bm))
        mad = np.median(res) * 1.4826 + 1e-6
        keep = res < max(3 * mad, 3.0)
        if keep.sum() < 8 or keep.all():
            break
        xs_a, ys_a = xs_a[keep], ys_a[keep]

    k, b = np.polyfit(xs_a, ys_a, 1)
    if abs(k) > 0.15:                    # 台面不可能过度倾斜
        k, b = 0.0, float(np.median(ys_a))
    return float(k), float(b)


def detect_baseline(gray_or_bgr: np.ndarray) -> tuple[float, float]:
    """
    自动检测台面基线, 返回 (k, b): y = k*x + b.

    步骤: 行亮度均值一阶差分找最强 暗->亮 跃变行 -> 附近亚像素精化 + 稳健拟合.
    """
    gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY) \
        if gray_or_bgr.ndim == 3 else gray_or_bgr
    h, _ = gray.shape

    smooth = np.convolve(gray.mean(axis=1), np.ones(31) / 31.0, mode='same')
    grad = np.diff(smooth)

    lo, hi = h // 5, h - h // 10
    base_y = lo + int(np.argmax(grad[lo:hi]))
    return _refine_baseline(gray, base_y)


# ===========================================================================
# 液滴轮廓提取
# ===========================================================================
def _baseline_y_map(baseline: Line, shape: tuple[int, int]) -> np.ndarray:
    """每个像素列对应的基线 y 值图 (h, w)."""
    h, w = shape
    xx = np.arange(w, dtype=float)[None, :].repeat(h, axis=0)
    if abs(baseline.unit[0]) > 1e-9:
        slope = baseline.unit[1] / baseline.unit[0]
        return baseline.origin[1] + slope * (xx - baseline.origin[0])
    return np.full((h, w), baseline.origin[1], dtype=float)


def _level_set_extremes(pts: np.ndarray, rz: np.ndarray,
                        bin_px: float = 2.0) -> np.ndarray:
    """逐 z 层取最左/最右点, 形成液滴两侧轮廓."""
    order = np.argsort(rz[1])
    rz_s, pts_s = rz[:, order], pts[:, order]
    z_max = rz_s[1].max()
    n_bins = max(1, int(z_max / bin_px))
    keep = np.zeros(pts_s.shape[1], dtype=bool)
    edges = np.linspace(0, z_max + 1e-6, n_bins + 1)
    idx_bounds = np.searchsorted(rz_s[1], edges)
    for s, e in zip(idx_bounds[:-1], idx_bounds[1:]):
        if e - s < 1:
            continue
        seg = rz_s[0][s:e]
        keep[s + int(np.argmin(seg))] = True
        keep[s + int(np.argmax(seg))] = True
    return pts_s[:, keep]


def _pick_drop_component(mask: np.ndarray, base_y_map: np.ndarray, *,
                         min_area: int = 150, min_height: int = 12,
                         max_width_frac: float = 0.55) -> Optional[np.ndarray]:
    """
    从二值掩码中筛选液滴连通域:
      判据 = 底部接近基线 + 高度足够 + 宽度远小于图宽;
      主成分与其 bbox 相邻的小碎片合并 (横向长条不参与).
    """
    h, w = mask.shape
    col_base = base_y_map[0]
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    best_label, best_score = 0, 0.0
    for i in range(1, n_lab):
        x0, y0, bw, bh, area = stats[i]
        if area < min_area or bh < min_height:
            continue
        if bw > max_width_frac * w:          # 台面棱线等横向长条
            continue
        cb = col_base[max(0, x0):min(w, x0 + bw)]
        gap = abs((y0 + bh) - cb.mean()) if cb.size else 999
        score = area * bh * (1.0 if gap < 12 else 0.15)
        if score > best_score:
            best_score, best_label = score, i
    if best_label == 0:
        return None

    main_bbox = stats[best_label][:4]
    keep_ids = {best_label}
    pad = 30
    for i in range(1, n_lab):
        if i in keep_ids or stats[i][4] < 40:
            continue
        sx0, sy0, sw, sh = stats[i][:4]
        if sw > 0.3 * w:
            continue
        if (sx0 < main_bbox[0] + main_bbox[2] + pad and
                sx0 + sw > main_bbox[0] - pad and
                sy0 < main_bbox[1] + main_bbox[3] + pad and
                sy0 + sh > main_bbox[1] - pad):
            keep_ids.add(i)
    return np.isin(labels, list(keep_ids)).astype(np.uint8) * 255


def _otsu_threshold(valid_pixels: np.ndarray) -> float:
    """仅在有效像素集合上计算 Otsu 阈值 (排除基线以下区域对直方图的干扰)."""
    vals = valid_pixels.ravel()
    if vals.size < 100:
        return -1.0
    hist = np.bincount(vals.astype(np.uint8), minlength=256).astype(float)
    total = hist.sum()
    sum_all = float((np.arange(256) * hist).sum())
    w_b = 0.0
    sum_b = 0.0
    best_thr, best_var = -1.0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var, best_thr = var_between, t
    return float(best_thr)


def extract_drop_profile(gray_enhanced: np.ndarray, baseline: Line, *,
                         thresh_frac: float = 0.22,
                         min_abs_grad: float = 12.0) -> np.ndarray:
    """
    提取液滴轮廓点 (2, N), 图像坐标.

    混合策略:
      A. 首选 Otsu 亮度分割 (暗背景亮液滴场景最稳)
      B. 回退 OpenDrop 风格梯度法 (Scharr 幅值 + 双阈值 + 连通域)
    """
    h, w = gray_enhanced.shape
    base_y_map = _baseline_y_map(baseline, (h, w))
    below_base = np.arange(h)[:, None] > (base_y_map + 2)

    # ---------- A. Otsu 亮度分割 ----------
    work = cv2.GaussianBlur(gray_enhanced, (5, 5), 0).copy()
    work[below_base] = 0                       # 基线下方置黑, 不参与阈值估计
    otsu_thr = _otsu_threshold(work[~below_base])
    if 20 <= otsu_thr <= 240:
        mask = ((work >= otsu_thr) * 255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        # 垂直开运算: 去掉贴着基线、高度仅几个像素的横向亮带,
        # 避免液滴与全图宽的噪声带连通成一个成分
        vker = np.ones((13, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vker)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        comp = _pick_drop_component(mask, base_y_map)
        if comp is not None and int((comp > 0).sum()) >= 300:
            py, px = np.nonzero(comp)
            pts = np.vstack([px, py]).astype(float)
            rz = baseline.to_rz(pts)
            if abs(rz[1].min()) > abs(rz[1].max()):
                rz[1] *= -1
            good = rz[1] > 1.5                 # 去掉贴着基线的噪声层
            if good.sum() >= 30:
                return _level_set_extremes(pts[:, good], rz[:, good])

    # ---------- B. 梯度法回退 ----------
    blur = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    dx = cv2.Scharr(blur, cv2.CV_16S, 1, 0).astype(np.float32)
    dy = cv2.Scharr(blur, cv2.CV_16S, 0, 1).astype(np.float32)
    mag = cv2.magnitude(dx, dy)

    mx = float(np.percentile(mag, 99.9))       # 用高分位数抗强反射边缘干扰
    if mx <= 0:
        return np.empty((2, 0), dtype=float)
    gmask = ((mag >= max(thresh_frac * mx, min_abs_grad)) * 255).astype(np.uint8)
    gmask[below_base] = 0
    gmask = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    comp = _pick_drop_component(gmask, base_y_map, min_area=80, min_height=8)
    if comp is None:
        return np.empty((2, 0), dtype=float)

    py, px = np.nonzero(comp)
    pts = np.vstack([px, py]).astype(float)
    if pts.shape[1] == 0:
        return pts

    rz = baseline.to_rz(pts)
    if abs(rz[1].min()) > abs(rz[1].max()):
        rz[1] *= -1
    return _level_set_extremes(pts, rz)


def _subpixel_refine(gray_blur: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    亚像素细化: 对每个轮廓点沿水平方向做 3 点抛物线插值逼近梯度极值位置.
    """
    if pts.shape[1] == 0:
        return pts
    gx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    h, w = gm.shape
    out = pts.copy()
    for j in range(pts.shape[1]):
        x, y = int(round(pts[0, j])), int(round(pts[1, j]))
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            ga, gb, gc = gm[y, x - 1], gm[y, x], gm[y, x + 1]
            denom = ga - 2 * gb + gc
            if denom < -1e-9:
                off = 0.5 * (ga - gc) / denom
                out[0, j] = x + float(np.clip(off, -1, 1))
    return out


# ===========================================================================
# 拟合: 圆弧 / 二次多项式
# ===========================================================================
def circle_fit(rz: np.ndarray, weights: Optional[np.ndarray] = None
               ) -> Optional[tuple[float, float, float, float]]:
    """
    最小二乘圆拟合 (Kåsa 初值 + 几何距离 Gauss-Newton 精修).
    返回 (rc, zc, R, rms) 或 None.
    """
    if rz.shape[1] < 5:
        return None
    r, z = rz[0], rz[1]

    A = np.column_stack([r, z, np.ones_like(r)])
    bb = r ** 2 + z ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
    except np.linalg.LinAlgError:
        return None
    rc0, zc0 = sol[0] / 2, sol[1] / 2
    R0 = math.sqrt(max(sol[2] + rc0 ** 2 + zc0 ** 2, 1e-9))
    if not (1.0 < R0 < 1e6):
        return None

    rc, zc, R = rc0, zc0, R0
    wt = np.ones_like(r) if weights is None else weights.copy()
    wt = wt / wt.max()
    for _ in range(30):
        d = np.hypot(r - rc, z - zc)
        if np.any(d < 1e-9):
            return None
        res = d - R
        J = np.column_stack([-(r - rc) / d, -(z - zc) / d, -np.ones_like(d)])
        JW = J * wt[:, None]
        try:
            upd, *_ = np.linalg.lstsq(JW.T @ J, -JW.T @ (res * wt), rcond=None)
        except np.linalg.LinAlgError:
            return None
        rc += upd[0]; zc += upd[1]; R += upd[2]
        if np.linalg.norm(upd) < 1e-10:
            break
    rms = float(np.sqrt(np.average((np.hypot(r - rc, z - zc) - R) ** 2,
                                   weights=wt)))
    return rc, zc, R, rms


def poly_fit_r(rz: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """
    二次多项式拟合 r = a*z^2 + b*z + c. 返回 (a, b, c, rms) 或 None.
    """
    if rz.shape[1] < 6:
        return None
    z, r = rz[1], rz[0]
    try:
        a, b, c = np.polyfit(z, r, 2)
    except np.linalg.LinAlgError:
        return None
    rms = float(np.sqrt(np.mean((np.polyval([a, b, c], z) - r) ** 2)))
    return float(a), float(b), float(c), rms


def _arc_tangent_angle(rc: float, zc: float, R: float, contact_r: float,
                       side_data_r: float) -> Optional[float]:
    """
    数值求圆弧在基线交点处的切线角 (弧度, 相对基线, 0~pi/2 区间的接触角语义).
    在接触点附近采样圆弧两点求弦方向, 避免象限讨论.
    """
    if abs(zc) >= R:
        return None
    phi_c = math.atan2(0.0 - zc, contact_r - rc)
    sign = 1.0 if side_data_r > contact_r else -1.0
    for dphi in (sign * 0.02, sign * 0.05, sign * 0.01):
        t = np.array([R * (math.cos(phi_c + dphi) - math.cos(phi_c)),
                      R * (math.sin(phi_c + dphi) - math.sin(phi_c))])
        if t[1] > 1e-9:
            return math.atan2(t[1], abs(t[0]))
    return None


def fit_one_side(rz_side: np.ndarray, *, prefer_circle: bool = True
                 ) -> FitResult:
    """
    单侧轮廓点 (基线坐标 2xN) 拟合并计算接触角.

    策略:
      - 取距接触点估计较近的近底部点 (局部形状决定接触角)
      - 圆弧与多项式并行拟合, 按 RMS 择优 (圆弧略优先, 物理意义明确)
    """
    fr = FitResult(n_points=rz_side.shape[1])
    if rz_side.shape[1] < 6:
        return fr

    r, z = rz_side[0], rz_side[1]

    # 接触点初值: z 最低点; 近底部门控: 距离 < 45% 特征尺度*2
    i_low = int(np.argmin(z))
    r_contact0 = r[i_low]
    scale = float(np.percentile(np.abs(r - r_contact0), 90) * 2 + 10)
    dist = np.hypot(r - r_contact0, z)
    near = dist < 0.45 * scale * 2
    if near.sum() < 8:
        near = np.ones(len(r), dtype=bool)
    sel = rz_side[:, near]
    wgt = 1.0 / (1.0 + dist[near])             # 越近权重越大

    cand: list[tuple[str, float, dict]] = []

    circ = circle_fit(sel, weights=wgt)
    if circ is not None:
        rc, zc, R, rms = circ
        if zc < 0 and abs(zc) < R:             # 圆心须在液滴内部 (z<0)
            l = math.sqrt(R * R - zc * zc)
            cr = min((rc - l, rc + l),
                     key=lambda c: float(np.abs(sel[0] - c).min()))
            ang = _arc_tangent_angle(rc, zc, R, cr, float(np.mean(sel[0])))
            if ang is not None:
                resid = np.abs(np.hypot(sel[0] - rc, sel[1] - zc) - R)
                cand.append(('circle', rms, {
                    'angle': ang, 'contact': np.array([cr, 0.0]),
                    'center': np.array([rc, zc]), 'radius': R}))

    poly = poly_fit_r(sel)
    if poly is not None:
        a, b, c, rms = poly
        if abs(a) < 50:
            ang = math.atan2(1.0, abs(b))      # tan(theta)=1/|dr/dz|
            cand.append(('poly', rms, {
                'angle': ang, 'contact': np.array([c, 0.0]),
                'center': None, 'radius': None}))

    if not cand:
        return fr

    # 选择: 默认圆弧优先, 但 RMS 劣化超过 1.5x 时用更优者
    circle_best = next((c for c in cand if c[0] == 'circle'), None)
    poly_best = next((c for c in cand if c[0] == 'poly'), None)
    if prefer_circle and circle_best is not None:
        chosen = circle_best
        if poly_best is not None and poly_best[1] < circle_best[1] / 1.5:
            chosen = poly_best
    else:
        chosen = min(cand, key=lambda t: t[1])

    method, rms, info = chosen
    fr.ok = True
    fr.method = method
    fr.angle_deg = math.degrees(info['angle'])
    fr.contact_rz = info['contact']
    fr.center_xy = info['center']
    fr.radius_px = info['radius']
    fr.rms = rms
    if rms > 5.0:                              # 拟合质量过差 -> 标记失败
        fr.rms = rms
        fr.angle_deg = None
        fr.ok = False
    return fr


# ===========================================================================
# 主测量流程
# ===========================================================================
def measure_drop(img_bgr: np.ndarray, filename: str = '', *,
                 thresh_frac: float = 0.22,
                 baseline_override: Optional[tuple[float, float]] = None,
                 render: bool = True) -> MeasureResult:
    """测量单张 BGR 图像的接触角, 返回 MeasureResult (含可视化图)."""
    res = MeasureResult(filename=filename)
    if img_bgr is None:
        res.error = '无法读取图像'
        return res

    h, w = img_bgr.shape[:2]
    enhanced = enhance_image(img_bgr)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    try:
        k, b = baseline_override if baseline_override is not None \
            else detect_baseline(gray)
    except Exception as e:
        res.error = f'基线检测失败: {e}'
        return res
    res.baseline_k, res.baseline_b = float(k), float(b)
    baseline = Line.from_slope(k, b, w)

    blur_g = cv2.GaussianBlur(gray, (5, 5), 0)
    pts = extract_drop_profile(gray, baseline, thresh_frac=thresh_frac)
    if pts.shape[1]:
        pts = _subpixel_refine(blur_g, pts)
    res.n_edge_points = int(pts.shape[1])

    def finish(err: Optional[str]) -> MeasureResult:
        res.error = err
        if render:
            res.annotated_bgr = _render(img_bgr, baseline, pts, res)
        return res

    if pts.shape[1] < 12:
        return finish(f'边缘点不足 ({pts.shape[1]})')

    rz = baseline.to_rz(pts)
    if abs(rz[1].min()) > abs(rz[1].max()):
        rz[1] *= -1
    rz = rz[:, rz[1] >= 0]
    if rz.shape[1] < 12:
        return finish('液滴不在基线上方')

    apex_i = int(np.argmax(rz[1]))
    apex_rz = rz[:, apex_i]
    res.apex_xy = baseline.to_xy(apex_rz)
    res.height_px = float(rz[1].max())

    left_m = rz[0] < apex_rz[0]
    res.left = fit_one_side(rz[:, left_m])
    res.right = fit_one_side(rz[:, ~left_m])

    for side in (res.left, res.right):
        if side.ok and side.contact_rz is not None:
            side.contact_xy = baseline.to_xy(side.contact_rz)
    if res.left.ok and res.right.ok:
        res.base_width_px = float(res.right.contact_rz[0] -
                                  res.left.contact_rz[0])

    la = res.left.angle_deg if res.left.ok else None
    ra = res.right.angle_deg if res.right.ok else None
    if la is None and ra is None:
        return finish('两侧拟合均失败')

    vals = [v for v in (la, ra) if v is not None]
    res.avg_angle = round(float(np.mean(vals)), 2)

    conf = 1.0
    for f_ in (res.left, res.right):
        if f_.ok:
            conf *= math.exp(-f_.rms / 4.0) * min(1.0, f_.n_points / 40.0)
        else:
            conf *= 0.3
    res.confidence = round(conf, 3)

    return finish(None)


# ===========================================================================
# 可视化渲染
# ===========================================================================
_COLORS = {
    'baseline': (0, 0, 255),
    'edge': (0, 255, 0),
    'arc': (0, 255, 255),
    'contact': (255, 0, 0),
}


def _render(img: np.ndarray, baseline: Line, pts: np.ndarray,
            res: MeasureResult) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(1.0, w / 1200)
    lw = max(1, round(2 * scale))
    fs = 0.7 * scale
    vis = img.copy()

    p0 = tuple(baseline.pt0.astype(int))
    p1 = tuple(baseline.pt1.astype(int))
    cv2.line(vis, p0, p1, _COLORS['baseline'], lw, cv2.LINE_AA)

    for x, y in pts.astype(int).T:
        cv2.circle(vis, (int(x), int(y)), max(1, round(scale)),
                   _COLORS['edge'], -1, cv2.LINE_AA)

    for f_, tag in ((res.left, 'L'), (res.right, 'R')):
        if not f_.ok or f_.contact_xy is None:
            continue
        cpt = f_.contact_xy
        cv2.drawMarker(vis, (int(cpt[0]), int(cpt[1])), _COLORS['contact'],
                       cv2.MARKER_CROSS, int(22 * scale), lw, cv2.LINE_AA)
        if f_.method == 'circle' and f_.center_xy is not None and f_.radius_px:
            c = f_.center_xy
            phi_c = math.atan2(cpt[1] - c[1], cpt[0] - c[0])
            span = min(math.pi / 3, 60.0 / max(f_.radius_px, 20))
            phis = np.linspace(phi_c - span, phi_c + span, 40)
            arc = np.stack([c[0] + f_.radius_px * np.cos(phis),
                            c[1] + f_.radius_px * np.sin(phis)], axis=1)
            arc = arc[arc[:, 1] < cpt[1] + 2]
            if len(arc) >= 2:
                cv2.polylines(vis, [arc.astype(np.int32).reshape(-1, 1, 2)],
                              False, _COLORS['arc'], lw, cv2.LINE_AA)
        elif f_.angle_deg is not None:
            th = math.radians(f_.angle_deg)
            direction = 1 if tag == 'R' else -1
            ln = 60 * scale
            p2 = (int(cpt[0] + direction * ln * math.cos(th)),
                  int(cpt[1] - ln * math.sin(th)))
            cv2.line(vis, (int(cpt[0]), int(cpt[1])), p2,
                     _COLORS['arc'], lw, cv2.LINE_AA)

    la = res.left.angle_deg if res.left.ok else None
    ra = res.right.angle_deg if res.right.ok else None

    def put(txt: str, color, y_off: int) -> int:
        nonlocal vis
        (tw, thh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
        ov = vis.copy()
        cv2.rectangle(ov, (8, y_off), (16 + tw, y_off + thh + 10), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.55, ov, 0.45, 0)
        cv2.putText(vis, txt, (12, y_off + thh), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, color, lw, cv2.LINE_AA)
        return y_off + int(32 * scale)

    y0 = 26
    if la is not None:
        m = res.left.method[0].upper() if res.left.method else '?'
        y0 = put(f"L {la:.1f} deg [{m} rms={res.left.rms:.2f}]",
                 (200, 220, 255), y0)
    if ra is not None:
        m = res.right.method[0].upper() if res.right.method else '?'
        y0 = put(f"R {ra:.1f} deg [{m} rms={res.right.rms:.2f}]",
                 (200, 255, 220), y0)
    if res.avg_angle is not None:
        put(f"Avg {res.avg_angle:.1f} deg  conf={res.confidence:.2f}",
            (120, 255, 255), y0)

    return vis
