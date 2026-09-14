"""Sobel-lemma · 接触线检测核心代码（多边缘算法模式）。

本步骤建立在 **Pre-process** 的输出之上：
    Pre-process 已经给出液滴掩码、包围盒、台面基线 (k, b)。
    Sobel-lemma 在这里用「可切换的边缘检测算法」重新识别液滴轮廓，
    并在轮廓与基线的交点处给出左右两个接触点（三线接触点在 2D 图中的投影）。

设计原则（独立实现，参考 Opendrop 的“先基线、再液滴、最后接触点”思路，
但不复制其代码）：

    1. 每个模式先产生一张边缘响应图（strength）和一张二值边缘图（edge）；
    2. 用预处理得到的液滴掩码提取“液滴轮廓带”（mask boundary band），
       从而把边缘限定在真实的液-气界面上，而不是台面纹理或倒影；
    3. 在轮廓带内寻找“距离基线最近、且边缘强度最强的像素”作为接触点，
       再做亚像素加权质心细化；
    4. 提取左右接触点之间的轮廓弧，输出接触线长度与接触跨度。

可切换模式：
    sobel            Sobel 梯度幅值
    scharr           Scharr 梯度幅值（对斜向边缘更敏感）
    laplacian        Laplacian 二阶导幅值
    sobel_laplacian  Sobel + Laplacian 归一化融合
    log              LoG（Marr-Hildreth 零交叉）
    canny            Canny 自适应阈值边缘

只做核心检测，不负责 Web 接口（app.py 负责）；所有函数都可单独调用。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    'EDGE_MODES',
    'THRESHOLD_MODES',
    'edge_strength',
    'edge_binary',
    'mask_band',
    'approximate_contacts',
    'detect_contact_line',
    'annotate_contact_line',
    'process_contact_line',
]

EDGE_MODES = {
    'sobel': 'Sobel 梯度',
    'scharr': 'Scharr 梯度',
    'laplacian': 'Laplacian 二阶导',
    'sobel_laplacian': 'Sobel + Laplacian',
    'log': 'LoG（Marr-Hildreth）',
    'canny': 'Canny 边缘',
}

THRESHOLD_MODES = {
    'otsu': 'Otsu 自适应阈值',
    'p88': '88% 分位阈值',
    'p95': '95% 分位阈值',
}


# --------------------------------------------------------------------------- #
# 基础几何：像素到基线的带符号距离
# --------------------------------------------------------------------------- #
def _zmap(k: float, b: float, w: int, h: int, invert: bool) -> np.ndarray:
    """z>0 表示“液滴一侧”。invert=False：液滴在基线上方；True：下方。"""
    xx = np.arange(w, dtype=np.float64)[None, :]
    yy = np.arange(h, dtype=np.float64)[:, None]
    dx = w - 1.0
    dy = k * (w - 1.0)
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n          # 沿基线方向的单位向量
    px, py = uy, -ux                 # 基线的法向量（指向“上方”）
    if invert:
        px, py = -px, -py
    return (px * xx + py * (yy - b)).astype(np.float32)


def _side(zm: np.ndarray, invert: bool, clearance: float) -> np.ndarray:
    """zmap 已经在 _zmap 里把「液滴一侧」统一成 z>0，所以这里不用再翻转。"""
    return zm > clearance


# --------------------------------------------------------------------------- #
# 边缘强度（每种模式的原始响应）
# --------------------------------------------------------------------------- #
def edge_strength(gray: np.ndarray, mode: str = 'sobel') -> np.ndarray:
    """返回 float32 边缘强度图（数值越大越像边缘）。"""
    if mode not in EDGE_MODES:
        raise ValueError(f'未知模式 {mode!r}，可选：{list(EDGE_MODES)}')

    g = cv2.GaussianBlur(gray, (5, 5), 0)

    if mode in ('sobel', 'scharr'):
        ks = 5 if mode == 'sobel' else -1
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ks)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ks)
        return cv2.magnitude(gx, gy)

    if mode == 'laplacian':
        return np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=5))

    if mode == 'sobel_laplacian':
        sobel = edge_strength(gray, 'sobel')
        lap = edge_strength(gray, 'laplacian')
        sn = _norm(sobel)
        ln = _norm(lap)
        return (0.62 * sn + 0.38 * ln).astype(np.float32)

    if mode == 'log':
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
        return np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=5))

    if mode == 'canny':
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    raise ValueError(mode)


def _norm(a: np.ndarray) -> np.ndarray:
    m = float(a.max()) if a.size else 1.0
    return (a / m).astype(np.float32) if m > 1e-9 else a.astype(np.float32)


def _zero_crossings(lap: np.ndarray) -> np.ndarray:
    """Laplacian 零交叉 → 二值边缘（Marr-Hildreth 风格）。"""
    h, w = lap.shape
    p = np.pad(lap, 1, mode='edge')
    s = np.sign(p)
    zc = np.zeros((h, w), dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = s[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            zc |= ((s[1:1 + h, 1:1 + w] * nb) < 0).astype(np.uint8)
    # 抑制极弱零交叉（纯噪声）
    zc &= (np.abs(lap) > 0.01 * float(np.abs(lap).max() + 1e-9)).astype(np.uint8)
    return zc


# --------------------------------------------------------------------------- #
# 二值边缘图
# --------------------------------------------------------------------------- #
def edge_binary(gray: np.ndarray, mode: str = 'sobel',
                threshold: str = 'otsu',
                mask: Optional[np.ndarray] = None) -> np.ndarray:
    """返回 0/255 二值边缘图。

    mask：可选，仅在这些像素内计算阈值并保留边缘（用于聚焦液滴区域）。
    """
    if mode == 'canny':
        med = float(np.median(gray))
        low = max(0, int(round(0.66 * med)))
        high = min(255, int(round(1.33 * med)))
        e = cv2.Canny(gray, low, high)
        if mask is not None:
            e &= (mask > 0).astype(np.uint8)
        return e

    if mode == 'log':
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
        lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=5)
        e = _zero_crossings(lap)
        if mask is not None:
            e &= (mask > 0).astype(np.uint8)
        return (e * 255).astype(np.uint8)

    strength = edge_strength(gray, mode)
    return _threshold_strength(strength, mask, threshold)


def _threshold_strength(strength: np.ndarray,
                        mask: Optional[np.ndarray],
                        method: str) -> np.ndarray:
    if mask is not None:
        m = (mask > 0)
        vals = strength[m]
    else:
        vals = strength.ravel()

    if vals.size < 64:
        return np.zeros_like(strength, dtype=np.uint8)

    if method == 'otsu':
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            thr = hi
        else:
            u8 = ((vals - lo) * (255.0 / (hi - lo))).astype(np.uint8)
            t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = lo + (t / 255.0) * (hi - lo)
    elif method.startswith('p'):
        pct = float(method[1:])
        thr = float(np.percentile(vals, pct))
    else:
        raise ValueError(f'未知阈值方式 {method!r}')

    e = (strength >= thr).astype(np.uint8)
    if mask is not None:
        e &= (mask > 0).astype(np.uint8)
    return (e * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 掩码轮廓带
# --------------------------------------------------------------------------- #
def mask_band(mask: np.ndarray, inner_r: int = 3, outer_r: int = 4) -> np.ndarray:
    """液滴掩码的边界带：外扩 outer_r、内缩 inner_r 之间的环。"""
    m = (mask > 0).astype(np.uint8)
    inner = cv2.erode(m, np.ones((2 * inner_r + 1, 2 * inner_r + 1), np.uint8))
    outer = cv2.dilate(m, np.ones((2 * outer_r + 1, 2 * outer_r + 1), np.uint8))
    return ((outer > 0) & (inner == 0))


def _clean(binary: np.ndarray, min_area: int = 60) -> np.ndarray:
    """去掉小连通块，返回 0/255。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), 8)
    out = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


# --------------------------------------------------------------------------- #
# 粗略接触点（由掩码估计，用作后续精细定位的种子）
# --------------------------------------------------------------------------- #
def approximate_contacts(mask: np.ndarray, k: float, b: float, invert: bool,
                         clearance: float = 3.0,
                         band: float = 8.0) -> Optional[Tuple]:
    """从掩码的“近基线带”估计左右接触点，返回 (left, right, center_x)。"""
    h, w = mask.shape
    zm = _zmap(k, b, w, h, invert)
    side = _side(zm, invert, clearance)
    near = side & (np.abs(zm) <= clearance + band) & (mask > 0)
    ys, xs = np.where(near)
    if xs.size < 2:
        ys, xs = np.where(mask > 0)
    if xs.size < 2:
        return None
    lx, rx = float(xs.min()), float(xs.max())
    return (lx, k * lx + b), (rx, k * rx + b), float((xs.min() + xs.max()) / 2.0)


# --------------------------------------------------------------------------- #
# 亚像素细化
# --------------------------------------------------------------------------- #
def _subpix(strength: np.ndarray, cx: int, cy: int, r: int = 2):
    h, w = strength.shape
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    win = strength[y0:y1, x0:x1].astype(np.float64)
    s = float(win.sum())
    if s <= 1e-9:
        return float(cx), float(cy)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((win * xx).sum() / s), float((win * yy).sum() / s)


def _contour_contacts(mask: np.ndarray, k: float, b: float, invert: bool,
                      clearance: float = 3.0, contact_band: float = 10.0):
    """在掩码外轮廓上找离基线最近的左右接触点（几何交点）。

    比边缘图局部搜索更稳健：直接基于预处理已保证质量的掩码拓扑，
    不受台面纹理/倒影等假边缘干扰。找不到时返回 None（不硬凑）。
    """
    h, w = mask.shape
    zm = _zmap(k, b, w, h, invert)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    xs = cnt[:, 0].astype(np.float64)
    ys = cnt[:, 1].astype(np.float64)
    z = zm[np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)]

    # 液滴侧 + 离基线接触带内
    near = (z > 0) & (z <= clearance + contact_band)
    if not near.any():
        near = z > 0          # 退而求其次：所有液滴侧轮廓点
    if not near.any():
        return None

    nxs, nys, nz = xs[near], ys[near], z[near]
    # 底部带：z 在 [zmin, zmin+0.5] 内的点（容忍 float32 噪声 / 像素离散），
    # 在其中取 x 最小/最大作为左右接触点
    zmin = float(nz.min())
    close = nz <= zmin + 0.5
    cx = nxs[close]
    cy = nys[close]
    if cx.size < 2:
        return None
    li = int(np.argmin(cx))  # x 最小 → 左接触点
    ri = int(np.argmax(cx))  # x 最大 → 右接触点
    if li == ri:
        return None
    return (float(cx[li]), float(cy[li])), (float(cx[ri]), float(cy[ri]))


def validate_contacts(left: Tuple[float, float], right: Tuple[float, float],
                      image_w: int, mask_w: float,
                      min_span_ratio: float = 0.3,
                      max_span_ratio: float = 0.9) -> Tuple[bool, List[str]]:
    """校验左右接触点合理性。返回 (ok, warnings)。"""
    warnings: List[str] = []
    span = math.hypot(right[0] - left[0], right[1] - left[1])
    if span <= 1.0:
        warnings.append('span 过小：左右接触点重合')
    if span > image_w * max_span_ratio:
        warnings.append('span 过大：超过图像宽度')
    return (len(warnings) == 0), warnings


# --------------------------------------------------------------------------- #
# 主流程：接触线检测
# --------------------------------------------------------------------------- #
def detect_contact_line(bgr: np.ndarray,
                        loc: Optional[Dict] = None,
                        *,
                        mode: str = 'sobel',
                        threshold: str = 'otsu',
                        contact_band: float = 10.0,
                        clearance: float = 3.0,
                        min_z: float = 5.0) -> Dict:
    """在预处理结果之上检测接触线。

    参数：
        bgr           —— 图像（与 loc 同尺寸）
        loc           —— Pre-process.locate_drop 的结果；None 时内部调用预处理
        mode          —— 边缘算法模式，见 EDGE_MODES
        threshold     —— 二值化阈值方式，见 THRESHOLD_MODES
        contact_band  —— 接触点搜索时允许的最大离基线距离（像素）
        clearance     —— 基线上方多远处不算“贴着台面”
        min_z         —— 提取液滴轮廓弧时，离基线多远处才纳入（排除台面段）

    返回 dict，包含左右接触点、接触线跨度、轮廓弧长、二值边缘图与可视化。
    """
    if mode not in EDGE_MODES:
        raise ValueError(f'未知模式 {mode!r}，可选：{list(EDGE_MODES)}')

    if loc is None:
        loc = _preprocess_locate(bgr)

    if not loc.get('ok'):
        return {'ok': False, 'mode': mode, 'error': loc.get('error', '预处理未找到液滴')}

    k, b = float(loc['k']), float(loc['b'])
    invert = bool(loc.get('invert', False))
    mask = loc['mask']
    if mask is None or mask.shape[:2] != bgr.shape[:2]:
        return {'ok': False, 'mode': mode, 'error': '掩码尺寸与图像不一致'}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    zm = _zmap(k, b, w, h, invert)
    side = _side(zm, invert, clearance)

    # 1) 聚焦液滴区域：掩码边界带 + 近基线接触带
    band_mask = mask_band(mask, inner_r=3, outer_r=4)
    contact_zone = side & (np.abs(zm) <= clearance + contact_band)
    focus = band_mask | (contact_zone & (mask > 0))

    # 2) 边缘图（模式决定）
    edge = edge_binary(gray, mode, threshold, mask=focus)
    strength = edge_strength(gray, mode)

    # 3) 接触点：优先用掩码轮廓几何交点（稳健），找不到再回退边缘图搜索
    contour_contacts = _contour_contacts(mask, k, b, invert, clearance, contact_band)
    if contour_contacts is not None:
        (lcx, lcy), (rcx, rcy) = contour_contacts
        left_edge = (lcx, lcy)
        right_edge = (rcx, rcy)
        conf_l = conf_r = 1.0
    else:
        approx = approximate_contacts(mask, k, b, invert, clearance, band=contact_band)
        if approx is None:
            return {'ok': False, 'mode': mode, 'error': '无法估计接触点位置'}
        (lx0, ly0), (rx0, ry0), cx = approx
        left_edge, conf_l = _refine_contact(edge, strength, mask, k, b, invert,
                                            lx0, clearance, contact_band)
        right_edge, conf_r = _refine_contact(edge, strength, mask, k, b, invert,
                                             rx0, clearance, contact_band)

    # 接触点是液滴轮廓与台面基线的交点：x 取轮廓上离基线最近的点，
    # y 落到基线上（三线接触点位于固体表面）。
    left = (left_edge[0], k * left_edge[0] + b)
    right = (right_edge[0], k * right_edge[0] + b)

    # 3.5) 有效性校验：span 不合理时判为失败，而不是硬返回一个错结果
    m_ys, m_xs = np.where(mask > 0)
    mask_w = float(m_xs.max() - m_xs.min()) if m_xs.size else 0.0
    valid, warnings = validate_contacts(left, right, w, mask_w)
    if not valid:
        return {'ok': False, 'mode': mode, 'error': '；'.join(warnings)}

    # 5) 液滴轮廓弧（液-气界面，排除台面段）
    outline = _extract_outline(edge, mask, zm, side, min_z)

    # 6) 接触线长度 / 跨度
    span = float(math.hypot(right[0] - left[0], right[1] - left[1]))
    arc_length, path = _arc_between(outline, mask, left, right, zm, invert, clearance)

    vis = annotate_contact_line(bgr, left, right, outline, path, k, b, invert, mode)

    return {
        'ok': True,
        'mode': mode,
        'mode_label': EDGE_MODES[mode],
        'threshold': threshold,
        'threshold_label': THRESHOLD_MODES.get(threshold, threshold),
        'edge': edge,
        'strength': strength,
        'outline': outline,
        'left': {'x': float(left[0]), 'y': float(left[1]),
                 'z': float(zm[int(np.clip(round(left[1]), 0, h - 1)),
                               int(np.clip(round(left[0]), 0, w - 1))]),
                 'conf': float(conf_l),
                 'edge': {'x': float(left_edge[0]), 'y': float(left_edge[1])}},
        'right': {'x': float(right[0]), 'y': float(right[1]),
                  'z': float(zm[int(np.clip(round(right[1]), 0, h - 1)),
                                int(np.clip(round(right[0]), 0, w - 1))]),
                  'conf': float(conf_r),
                  'edge': {'x': float(right_edge[0]), 'y': float(right_edge[1])}},
        'span': span,
        'arc_length': arc_length,
        'baseline': {'k': k, 'b': b, 'invert': invert},
        'vis': vis,
    }


def _preprocess_locate(bgr: np.ndarray) -> Dict:
    """懒加载 Pre-process 的定位函数，避免本模块强依赖。"""
    try:
        from preprocess import locate_drop  # type: ignore
    except ImportError:
        import os
        import sys
        parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pre-process'))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from preprocess import locate_drop  # type: ignore
    return locate_drop(bgr)


def _refine_contact(edge: np.ndarray, strength: np.ndarray, mask: np.ndarray,
                    k: float, b: float, invert: bool, x0: float,
                    clearance: float, contact_band: float):
    """在边缘图的轮廓带内，找离基线最近且强度最高的点。

    允许搜索到基线下侧一小段（below），这样轮廓真正穿过基线处也能被捕捉，
    而不是永远停在掩码下缘（掩码本身只保留 clearance 以上的像素）。
    """
    h, w = edge.shape
    zm = _zmap(k, b, w, h, invert)
    band_mask = mask_band(mask, inner_r=3, outer_r=5)
    below = 6.0
    near = (zm >= -below) & (zm <= clearance + contact_band)

    zone = (edge > 0) & band_mask & near
    ys, xs = np.where(zone)
    if xs.size == 0:
        # 退而求其次：不要求落在轮廓带内，只要在近基线一带
        zone = (edge > 0) & near
        ys, xs = np.where(zone)
    if xs.size == 0:
        return (x0, k * x0 + b), 0.0

    keep = np.abs(xs - x0) <= 48
    if keep.any():
        ys, xs = ys[keep], xs[keep]

    zz = np.abs(zm[ys, xs])
    order = np.argsort(zz, kind='stable')
    top = order[:min(8, order.size)]
    best = int(top[0])
    bs = float(strength[ys[best], xs[best]])
    for i in top:
        s = float(strength[ys[i], xs[i]])
        if s > bs:
            bs, best = s, int(i)

    px, py = _subpix(strength, int(xs[best]), int(ys[best]))
    return (px, py), bs


def _extract_outline(edge: np.ndarray, mask: np.ndarray, zm: np.ndarray,
                     side: np.ndarray, min_z: float) -> np.ndarray:
    """液滴轮廓弧：边缘图 ∩ 掩码边界带 ∩ 液滴侧 ∩ 离基线足够远。

    先做一次小闭运算，把轻微断裂的边缘连成连续的轮廓弧。
    """
    band_mask = mask_band(mask, inner_r=3, outer_r=4)
    raw = (edge > 0) & band_mask & side & (np.abs(zm) > min_z)
    raw = cv2.morphologyEx((raw * 255).astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((5, 5), np.uint8))
    return _clean(raw)


def _arc_between(outline: np.ndarray, mask: np.ndarray,
                 left: Tuple[float, float], right: Tuple[float, float],
                 zm: np.ndarray, invert: bool, clearance: float):
    """左右接触点之间的轮廓弧长。

    优先用模式边缘图上的连通路径；若边缘不连续，回退到掩码外轮廓的穹顶段。
    """
    path = _edge_path(outline, left, right)
    if path is not None and len(path) >= 3:
        return _polyline_len(path), path

    path = _refined_dome_path(mask, outline, left, right, zm, invert, clearance)
    if path is not None and len(path) >= 3:
        return _polyline_len(path), path

    path = _mask_dome_path(mask, left, right, zm, invert, clearance)
    if path is not None and len(path) >= 3:
        return _polyline_len(path), path
    return 0.0, None


def _edge_path(outline: np.ndarray, p1, p2):
    """在轮廓二值图上，找连接 p1、p2 的 8 邻接最短路径。"""
    o = (outline > 0).astype(np.uint8)
    if o.sum() < 3:
        return None
    a = _snap(o, p1)
    b = _snap(o, p2)
    if a is None or b is None or a == b:
        return None

    n, lab, _, _ = cv2.connectedComponentsWithStats(o, 8)
    if lab[a[1], a[0]] == 0 or lab[a[1], a[0]] != lab[b[1], b[0]]:
        return None

    # BFS 最短路径（8 邻接），返回按顺序排列的像素点
    from collections import deque
    h, w = o.shape
    prev = np.full((h, w), -1, dtype=np.int64)
    visited = np.zeros((h, w), dtype=bool)
    q = deque([a])
    visited[a[1], a[0]] = True
    dirs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    found = False
    while q:
        x, y = q.popleft()
        if (x, y) == b:
            found = True
            break
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and o[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                prev[ny, nx] = y * w + x
                q.append((nx, ny))
    if not found:
        return None

    pts = [b]
    x, y = b
    while (x, y) != a:
        p = prev[y, x]
        if p < 0:
            return None
        x, y = p % w, p // w
        pts.append((x, y))
    pts.reverse()
    return np.array(pts, dtype=np.float64)


def _snap(o: np.ndarray, p, radius: int = 20):
    """把点吸附到最近的轮廓像素。"""
    h, w = o.shape
    x0, y0 = int(round(p[0])), int(round(p[1]))
    ys, xs = np.where(o > 0)
    if xs.size == 0:
        return None
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    i = int(np.argmin(d2))
    if d2[i] > radius * radius:
        return None
    return (int(xs[i]), int(ys[i]))


def _refined_dome_path(mask: np.ndarray, outline: np.ndarray,
                       left, right, zm, invert, clearance,
                       snap_r: int = 8):
    """掩码轮廓 + 模式边缘图的混合路径。

    掩码轮廓保证拓扑连通（不会被边缘断裂影响），再把每个点吸附到最近边缘像素，
    从而让轮廓长度也反映所选边缘算法的定位结果。
    """
    path = _mask_dome_path(mask, left, right, zm, invert, clearance)
    if path is None or len(path) < 3:
        return None
    h, w = outline.shape
    refined = np.array(path, dtype=np.float64)
    for i, (x, y) in enumerate(path):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - snap_r), min(w, xi + snap_r + 1)
        y0, y1 = max(0, yi - snap_r), min(h, yi + snap_r + 1)
        win = outline[y0:y1, x0:x1]
        idx = np.argwhere(win > 0)
        if idx.size == 0:
            continue
        dys, dxs = idx[:, 0], idx[:, 1]
        d2 = (dxs - (xi - x0)) ** 2 + (dys - (yi - y0)) ** 2
        j = int(np.argmin(d2))
        refined[i] = (x0 + dxs[j], y0 + dys[j])
    return refined


def _mask_dome_path(mask: np.ndarray, left, right, zm, invert, clearance):
    """掩码外轮廓中，左右接触点之间的穹顶段（液滴侧）。"""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if cnt.shape[0] < 6:
        return None

    dl = np.linalg.norm(cnt - np.array(left), axis=1)
    dr = np.linalg.norm(cnt - np.array(right), axis=1)
    il, ir = int(np.argmin(dl)), int(np.argmin(dr))
    if il == ir:
        return None
    if il > ir:
        il, ir = ir, il

    seg1 = cnt[il:ir + 1]
    seg2 = np.concatenate([cnt[ir:], cnt[:il + 1]], axis=0)

    def dome_score(seg):
        if seg.shape[0] < 2:
            return -1.0
        ys = np.clip(seg[:, 1].astype(int), 0, zm.shape[0] - 1)
        xs = np.clip(seg[:, 0].astype(int), 0, zm.shape[1] - 1)
        z = zm[ys, xs]
        return float(np.mean(np.abs(z)))

    return seg1 if dome_score(seg1) >= dome_score(seg2) else seg2


def _polyline_len(pts) -> float:
    if pts is None or len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def annotate_contact_line(bgr: np.ndarray, left, right,
                          outline: np.ndarray, path,
                          k: float, b: float, invert: bool,
                          mode: str) -> np.ndarray:
    """画出边缘轮廓、左右接触点、基线与接触跨度。"""
    vis = bgr.copy()
    h, w = vis.shape[:2]

    # 半透明显示检测到的轮廓弧
    if outline is not None and outline.shape[:2] == (h, w):
        vis[outline > 0] = (vis[outline > 0] * 0.35 + np.array([0, 255, 120]) * 0.65).astype(np.uint8)

    # 路径弧（细绿线）
    if path is not None and len(path) >= 2:
        pts = path.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, (0, 255, 140), 2, cv2.LINE_AA)

    # 基线
    y0 = int(round(b))
    y1 = int(round(k * (w - 1) + b))
    cv2.line(vis, (0, y0), (w - 1, y1), (80, 80, 255), 2, cv2.LINE_AA)

    # 接触跨度（左右接触点连线）
    cv2.line(vis, (int(round(left[0])), int(round(left[1]))),
             (int(round(right[0])), int(round(right[1]))), (255, 220, 60), 2, cv2.LINE_AA)

    # 接触点
    for p, tag in ((left, 'L'), (right, 'R')):
        cx, cy = int(round(p[0])), int(round(p[1]))
        cv2.circle(vis, (cx, cy), 6, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (cx, cy), (0, 200, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(vis, tag, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 220, 255), 2, cv2.LINE_AA)

    s = max(1.0, w / 1200.0)
    txt = [
        f'edge mode: {mode}',
        f'L=({left[0]:.1f}, {left[1]:.1f})  R=({right[0]:.1f}, {right[1]:.1f})',
        f'span: {math.hypot(right[0]-left[0], right[1]-left[1]):.1f} px',
    ]
    yy = int(34 * s)
    for t in txt:
        cv2.putText(vis, t, (int(14 * s), yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65 * s, (245, 245, 245), max(1, int(1.5 * s)), cv2.LINE_AA)
        yy += int(26 * s)
    return vis


# --------------------------------------------------------------------------- #
# 供 UI 使用的完整流水线：预处理 → 接触线检测
# --------------------------------------------------------------------------- #
def process_contact_line(bgr: np.ndarray, *,
                         use_sr: bool = False,
                         sr_scale: float = 3.0,
                         mode: str = 'sobel',
                         threshold: str = 'otsu',
                         contact_band: float = 10.0) -> Dict:
    """预处理 + 接触线检测流水线，输出可直接交给 UI 展示的各阶段图像。

    返回：
        stages: [(名称, BGR图, 描述), ...]
        result: detect_contact_line 的结果
        ok: bool
    """
    try:
        import preprocess as pp  # 由 app.py 保证路径，或本模块懒加载
    except ImportError:
        import os as _os
        import sys as _sys
        _parent = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', 'Pre-process'))
        if _parent not in _sys.path:
            _sys.path.insert(0, _parent)
        import preprocess as pp  # type: ignore  # noqa: F811

    stages: List[Tuple[str, np.ndarray, str]] = []
    h, w = bgr.shape[:2]
    stages.append(('原图', bgr.copy(), f'{w}×{h}'))

    work = bgr
    sr_ok = False
    if use_sr and sr_scale > 1:
        sr_ok = pp.sr_available()
        work = pp.super_resolve(work, sr_scale)
        tag = '超分辨率' if sr_ok else '超分辨率（经典回退）'
        stages.append((tag, work.copy(),
                       f'{work.shape[1]}×{work.shape[0]}，scale={int(round(sr_scale))}'))

    loc = pp.locate_drop(work)
    stages.append(('液滴定位（预处理）', pp.annotate(work, loc),
                   str(loc['bbox']) if loc.get('ok') else '定位失败'))

    result = detect_contact_line(work, loc, mode=mode, threshold=threshold,
                                 contact_band=contact_band)

    # 边缘图单独展示（转为彩色便于看清）
    if result.get('ok'):
        edge = result['edge']
        edge_vis = np.zeros((*edge.shape, 3), dtype=np.uint8)
        edge_vis[edge > 0] = (120, 255, 120)
        stages.append((f'边缘图 · {result["mode_label"]}', edge_vis,
                       f'阈值：{result["threshold_label"]}'))
        stages.append(('接触线检测结果', result['vis'],
                       f"L({result['left']['x']:.1f},{result['left']['y']:.1f}) "
                       f"R({result['right']['x']:.1f},{result['right']['y']:.1f})"))
    else:
        stages.append(('接触线检测失败', work.copy(), result.get('error', '')))

    return {'stages': stages, 'result': result, 'sr_active': sr_ok, 'ok': result.get('ok', False)}
