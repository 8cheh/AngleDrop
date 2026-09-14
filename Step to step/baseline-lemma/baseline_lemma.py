"""baseline-lemma · 基线识别核心代码（多算法模式）。

本目录是 **AngleDrop「Step to step」的第 3 步 —— baseline-lemma 基线识别**：
在给定液滴图像中，用**可切换的算法**识别台面/衬底基线

        y = k * x + b

（三线接触点所在的固体表面直线）。它建立在第 1 步 Pre-process 的输出之上，
但基线识别本身独立实现：Pre-process 的 `locate_drop` 里已经内置了一条基线，
本步骤用更多、更明确的算法重新识别，便于横向对比不同算子的表现。

设计原则（独立实现，不复制 Opendrop 代码）：

    1. 基线是近水平的直线，且是「液滴侧 / 衬底侧」之间的一条强边缘；
    2. 每种模式独立给出「候选直线 + 置信度 + 证据（点/线段/剖面）」，互不依赖；
    3. 检测结果统一输出 (k, b)、斜率角、液滴侧方向（invert）、RMSE 与置信度；
    4. 可视化分「证据图」与「最终叠加图」两层，便于检查每个算法到底“看”到了什么。

可切换模式（BASELINE_MODES）：

    sobel_ransac    Sobel 行梯度剖面 + 逐列亚像素 + RANSAC（经典梯度法）
    hough           Canny + HoughLinesP 直线投票（霍夫变换法）
    otsu_column     Otsu 列剖面界面定位 + RANSAC（Otsu 阈值法）
    fitline         Canny 边缘点 + cv2.fitLine（Huber 稳健最小二乘）
    edge_ransac     Canny 边缘点 + 显式 RANSAC 拟合（随机采样一致）
    pca             Canny 边缘点 + PCA 主方向拟合（主成分分析）

只做核心检测，不负责 Web 接口（app.py 负责）；所有函数都可单独调用。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    'BASELINE_MODES',
    'detect_baseline',
    'annotate_baseline',
    'process_baseline',
]

BASELINE_MODES = {
    'sobel_ransac': 'Sobel 行梯度 + RANSAC',
    'hough': 'Hough 直线投票',
    'otsu_column': 'Otsu 列剖面界面',
    'fitline': '边缘点 Huber 拟合 (fitLine)',
    'edge_ransac': '边缘点 RANSAC',
    'pca': '边缘点 PCA 主方向',
}


# --------------------------------------------------------------------------- #
# 基础工具：灰度 / Sobel / 行剖面 / RANSAC / 几何
# --------------------------------------------------------------------------- #
def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _sobel_y(gray: np.ndarray) -> np.ndarray:
    """竖向 Sobel（dI/dy）。正值 = 越往下越亮（暗上亮下）。"""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)


def _row_profile_abs(gray: np.ndarray) -> np.ndarray:
    """|Sobel_y| 沿 x 求和得到的行剖面（每行的总水平边缘强度）。"""
    sob = _sobel_y(gray)
    prof = np.abs(sob).sum(axis=1).astype(np.float64)
    return np.convolve(prof, np.ones(15) / 15.0, mode='same')


def _coarse_row(gray: np.ndarray) -> int:
    """最强水平边缘所在行（用于给「拟合类」算法一个粗定位带）。"""
    prof = _row_profile_abs(gray)
    return int(np.argmax(prof))


def _candidate_rows(gray: np.ndarray, max_cands: int = 8) -> List[int]:
    """行剖面峰值候选行，按强度降序，且彼此间隔 > 30 行。"""
    prof = _row_profile_abs(gray)
    order = np.argsort(prof)[::-1]
    rows: List[int] = []
    for r in order:
        r = int(r)
        if not rows or all(abs(r - q) > 30 for q in rows):
            rows.append(r)
        if len(rows) >= max_cands:
            break
    return rows


def _subpixel_peak_cols(sob: np.ndarray, row: float, win: int = 40,
                        polarity: float = 1.0, step: int = 3
                        ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """在 `row` 附近 ±win 行内，逐列（每 step 像素采一列）找竖向边缘峰值并做
    抛物线亚像素细化。返回 (xs, ys, peak)。"""
    h, w = sob.shape
    y0 = int(max(0, round(row) - win))
    y1 = int(min(h, round(row) + win))
    if y1 - y0 < 8:
        return None

    band = (sob * polarity)[y0:y1, :]
    xs = np.arange(0, w, step)
    sub = band[:, xs]
    idx = np.argmax(sub, axis=0)
    cols = np.arange(sub.shape[1])
    peak = sub[idx, cols]

    a = sub[np.clip(idx - 1, 0, sub.shape[0] - 1), cols]
    c = sub[np.clip(idx + 1, 0, sub.shape[0] - 1), cols]
    den = a - 2.0 * peak + c
    den_safe = np.where(np.abs(den) > 1e-6, den, 1.0)
    off = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / den_safe, 0.0)
    ys = y0 + idx + np.clip(off, -1.0, 1.0)
    return xs.astype(np.float64), ys.astype(np.float64), peak.astype(np.float64)


def _ransac_line(P: np.ndarray, max_slope: float = 0.2, thresh: float = 4.0,
                 iters: int = 400, seed: int = 0
                 ) -> Optional[Tuple[float, float, np.ndarray]]:
    """对点集 P (N,2) 做 RANSAC 近水平直线拟合，返回 (k, b, inliers)。"""
    if P is None or len(P) < 4:
        return None
    X = P[:, 0].astype(np.float64)
    Y = P[:, 1].astype(np.float64)
    n = X.size
    xspan = float(X.max() - X.min())
    if xspan < 20:
        return None

    rng = np.random.default_rng(seed)
    best_n = 0
    best_in: Optional[np.ndarray] = None
    best_k = best_b = None
    for _ in range(iters):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        if abs(X[i] - X[j]) < 0.2 * xspan:
            continue
        k = (Y[j] - Y[i]) / (X[j] - X[i])
        if abs(k) > max_slope:
            continue
        b = Y[i] - k * X[i]
        inl = np.abs(Y - (k * X + b)) < thresh
        cnt = int(inl.sum())
        if cnt > best_n:
            best_n = cnt
            best_in = inl
            best_k, best_b = float(k), float(b)

    if best_in is None or best_n < 6:
        return None
    k, b = np.polyfit(X[best_in], Y[best_in], 1)
    if abs(k) > max_slope:
        return None
    inl = np.abs(Y - (k * X + b)) < thresh
    if inl.sum() < 6:
        return None
    return float(k), float(b), inl


def _inliers_to_line(P: np.ndarray, k: float, b: float,
                     thresh: float = 4.0) -> np.ndarray:
    if P is None or len(P) == 0:
        return np.zeros(0, dtype=bool)
    X, Y = P[:, 0], P[:, 1]
    return np.abs(Y - (k * X + b)) < thresh


def _rmse(P: np.ndarray, k: float, b: float) -> Optional[float]:
    if P is None or len(P) == 0:
        return None
    X, Y = P[:, 0], P[:, 1]
    return float(np.sqrt(np.mean((Y - (k * X + b)) ** 2)))


def _line_endpoints(k: float, b: float, w: int, h: int) -> Tuple[Tuple[float, float],
                                                                 Tuple[float, float]]:
    """直线与图像左右边界的交点（可视化用）。"""
    return (0.0, float(b)), (float(w - 1), float(k * (w - 1) + b))


def _line_polarity(sob_y: np.ndarray, k: float, b: float, w: int, h: int,
                   band: int = 6) -> float:
    """沿直线采样 Sobel_y 的符号，估计边缘极性：+1=暗上亮下，-1=亮上暗下。

    为容忍直线拟合偏差，在每个采样点上下 ±band 的小窗口内取 |Sobel_y| 最强的
    位置的值，而不是只取直线正上的单点。
    """
    xs = np.linspace(0.0, w - 1.0, 40)
    ys = k * xs + b
    vals: List[float] = []
    for x, y in zip(xs, ys):
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        y0 = max(0, yi - band)
        y1 = min(h, yi + band + 1)
        win = sob_y[y0:y1, xi]
        if win.size == 0:
            continue
        vals.append(float(win[int(np.argmax(np.abs(win)))]))
    if len(vals) < 8:
        return 1.0
    return 1.0 if float(np.mean(vals)) > 0 else -1.0


# --------------------------------------------------------------------------- #
# 各算法模式
# --------------------------------------------------------------------------- #
def _mode_sobel_ransac(gray: np.ndarray, max_slope: float, seed: int,
                       max_cands: int = 8) -> Optional[Dict]:
    """Sobel 行梯度剖面 + 逐列亚像素 + RANSAC。

    行剖面找到候选行后，对每个候选行分别用正/负极性提取逐列峰值点，
    RANSAC 拟合近水平线，按「内点数 × 平均边缘强度 × 横跨覆盖率²」打分；
    覆盖率惩罚用于排除只覆盖局部短弧的假基线（如液滴顶部轮廓）。
    """
    h, w = gray.shape
    sob = _sobel_y(gray)
    prof = np.abs(sob).sum(axis=1).astype(np.float64)
    prof = np.convolve(prof, np.ones(15) / 15.0, mode='same')

    best = None
    for row in _candidate_rows(gray, max_cands):
        for pol in (1.0, -1.0):
            peaks = _subpixel_peak_cols(sob, row, win=40, polarity=pol)
            if peaks is None:
                continue
            xs, ys, peak = peaks
            thr = max(8.0, float(np.percentile(peak, 50)))
            strong = peak > thr
            if strong.sum() < 12:
                strong = np.ones(peak.shape, dtype=bool)
            P = np.stack([xs[strong], ys[strong]], axis=1)
            fit = _ransac_line(P, max_slope=max_slope, thresh=3.0,
                               iters=250, seed=seed)
            if fit is None:
                continue
            k, b, inl = fit
            if inl.sum() < 8:
                continue
            strength = float(peak[strong][inl].mean()) if inl.any() else 0.0
            if strength < 1.0:
                # 边缘强度为零（纯空白/纯噪声图），视为无效候选
                continue
            # 基线应横跨整幅图：内点 x 跨度越大越可信（排除液滴穹顶等短弧边缘）。
            # coverage 取平方，强惩罚只覆盖局部短弧的假基线（如液滴顶部轮廓）。
            xspan = float(P[inl, 0].max() - P[inl, 0].min()) if inl.any() else 0.0
            coverage = xspan / max(1.0, float(w))
            score = float(inl.sum()) * strength * (0.05 + coverage * coverage)
            if best is None or score > best[0]:
                best = (score, float(k), float(b), pol, P, inl, row, prof)

    if best is None:
        return None
    score, k, b, pol, P, inl, row, prof = best
    return {'k': k, 'b': b, 'polarity': pol, 'score': score,
            'points': P, 'inliers': inl, 'segments': None,
            'evidence': {'row': row, 'profile': prof}}


def _mode_hough(gray: np.ndarray, max_slope: float) -> Optional[Dict]:
    """Canny + HoughLinesP 直线投票：筛选近水平线段，按长度/边缘密度/位置打分。"""
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 50, 150)
    min_len = max(60, int(w * 0.12))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70,
                            minLineLength=min_len, maxLineGap=30)
    if lines is None:
        return None
    lines = lines.reshape(-1, 4)

    segs = []
    for x1, y1, x2, y2 in lines.tolist():
        if x2 == x1:
            continue
        k = (y2 - y1) / (x2 - x1)
        if abs(k) > max_slope:
            continue
        segs.append((int(x1), int(y1), int(x2), int(y2), float(k)))

    if not segs:
        return None

    scored = []
    for x1, y1, x2, y2, k in segs:
        length = math.hypot(x2 - x1, y2 - y1)
        b = y1 - k * x1
        n_sample = max(20, min(80, int(length) // 2))
        xs = np.linspace(min(x1, x2), max(x1, x2), n_sample)
        ys = k * xs + b
        valid = (ys >= 0) & (ys < h - 1)
        if valid.any():
            dens = float(edges[np.clip(ys[valid].astype(int), 0, h - 1),
                               np.clip(xs[valid].astype(int), 0, w - 1)].mean()) / 255.0
        else:
            dens = 0.0
        cy = (y1 + y2) / 2.0
        center = 1.0 - min(1.0, abs(cy - h * 0.5) / (h * 0.5))
        score = length * (0.5 + dens) * (0.6 + 0.4 * center)
        scored.append((score, k, b, (x1, y1, x2, y2), dens))

    scored.sort(key=lambda t: -t[0])
    score, k, b, seg, dens = scored[0]
    top_segs = [s for s in [seg] + [t[3] for t in scored[1:8]]]
    return {'k': float(k), 'b': float(b), 'polarity': None, 'score': float(score),
            'points': None, 'inliers': None, 'segments': top_segs,
            'evidence': {'edges': edges, 'best_density': float(dens)}}


def _mode_otsu_column(gray: np.ndarray, max_slope: float, seed: int,
                      step: int = 4) -> Optional[Dict]:
    """Otsu 列剖面界面定位 + RANSAC。

    逐列做 Otsu 二值化，找到列内亮度两类之间的分界（候选过渡行），
    取其中竖向边缘最强的一处作为衬底表面点，收集所有列的点后用 RANSAC 拟合。
    """
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    sob = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)

    xs = np.arange(0, w, step)
    pts: List[Tuple[float, float]] = []
    for x in xs:
        col8 = g[:, x]
        thr, _ = cv2.threshold(col8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (col8 >= thr).astype(np.int8)
        trans = np.where(np.diff(binary) != 0)[0]

        best_t, best_s = None, -1.0
        for t in trans:
            if t < 3 or t > h - 4:
                continue
            s = float(np.abs(sob[max(0, t - 2):t + 3, x]).max())
            if s > best_s:
                best_s, best_t = s, int(t)
        if best_t is None:
            continue

        # 亚像素细化（基于 |Sobel_y| 的抛物线）
        t = best_t
        y0, y1 = max(0, t - 1), min(h, t + 2)
        seg = np.abs(sob[y0:y1, x]).astype(np.float64)
        i = int(np.argmax(seg))
        left = seg[max(0, i - 1)]
        center = seg[i]
        right = seg[min(len(seg) - 1, i + 1)]
        den = left - 2.0 * center + right
        off = 0.5 * (left - right) / den if abs(den) > 1e-6 else 0.0
        y = (y0 + i) + float(np.clip(off, -1.0, 1.0))
        pts.append((float(x), float(y)))

    if len(pts) < 8:
        return None
    P = np.array(pts, dtype=np.float64)
    fit = _ransac_line(P, max_slope=max_slope, thresh=4.0, iters=300, seed=seed)
    if fit is None:
        return None
    k, b, inl = fit
    if inl.sum() < 6:
        return None

    xi = np.clip(P[:, 0].astype(int), 0, w - 1)
    yi = np.clip(P[:, 1].astype(int), 0, h - 1)
    strength = np.abs(sob[yi, xi])
    score = float(inl.sum()) * float(strength[inl].mean()) if inl.any() else 0.0
    return {'k': float(k), 'b': float(b), 'polarity': None, 'score': score,
            'points': P, 'inliers': inl, 'segments': None,
            'evidence': {'profile': _row_profile_abs(gray)}}


def _edge_points_for_fit(gray: np.ndarray, band: int = 60
                         ) -> Tuple[Optional[np.ndarray], np.ndarray, int]:
    """给「拟合类」算法收集候选边缘点：粗定位行 ±band 内的 Canny 边缘点。"""
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 50, 150)
    row = _coarse_row(gray)
    y0 = max(0, row - band)
    y1 = min(h, row + band)
    ys, xs = np.where(edges[y0:y1, :] > 0)
    if xs.size < 6:
        return None, edges, row
    return np.stack([xs.astype(np.float64), (ys + y0).astype(np.float64)], axis=1), edges, row


def _mode_fitline(gray: np.ndarray, max_slope: float, band: int) -> Optional[Dict]:
    """Canny 边缘点 + cv2.fitLine（DIST_HUBER 稳健最小二乘）。"""
    P, edges, row = _edge_points_for_fit(gray, band)
    if P is None or len(P) < 6:
        return None
    line = cv2.fitLine(P.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.ravel()
    if abs(vx) < 1e-9:
        return None
    k = float(vy / vx)
    if abs(k) > max_slope:
        return None
    b = float(y0 - k * x0)
    inl = _inliers_to_line(P, k, b, thresh=4.0)
    if inl.sum() < 6:
        return None
    rmse = _rmse(P[inl], k, b)
    score = float(inl.sum()) / (1.0 + (rmse if rmse is not None else 4.0))
    return {'k': k, 'b': b, 'polarity': None, 'score': score,
            'points': P, 'inliers': inl, 'segments': None,
            'evidence': {'row': row, 'edges': edges}}


def _mode_edge_ransac(gray: np.ndarray, max_slope: float, band: int,
                      seed: int) -> Optional[Dict]:
    """Canny 边缘点 + 显式 RANSAC 拟合（随机采样一致性）。"""
    P, edges, row = _edge_points_for_fit(gray, band)
    if P is None or len(P) < 6:
        return None
    fit = _ransac_line(P, max_slope=max_slope, thresh=4.0, iters=400, seed=seed)
    if fit is None:
        return None
    k, b, inl = fit
    if inl.sum() < 6:
        return None
    rmse = _rmse(P[inl], k, b)
    score = float(inl.sum()) / (1.0 + (rmse if rmse is not None else 4.0))
    return {'k': float(k), 'b': float(b), 'polarity': None, 'score': score,
            'points': P, 'inliers': inl, 'segments': None,
            'evidence': {'row': row, 'edges': edges}}


def _mode_pca(gray: np.ndarray, max_slope: float, band: int) -> Optional[Dict]:
    """Canny 边缘点 + PCA 主方向拟合：主方向即基线方向，直线过点集质心。"""
    P, edges, row = _edge_points_for_fit(gray, band)
    if P is None or len(P) < 6:
        return None
    mu = P.mean(axis=0)
    centered = P - mu
    cov = centered.T @ centered / len(P)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    if abs(v[0]) < 1e-9:
        return None
    k = float(v[1] / v[0])
    if abs(k) > max_slope:
        return None
    b = float(mu[1] - k * mu[0])
    inl = _inliers_to_line(P, k, b, thresh=4.0)
    if inl.sum() < 6:
        return None

    lmax = float(evals.max())
    lmin = float(evals.min()) + 1e-9
    collinear = lmax / lmin
    rmse = _rmse(P[inl], k, b)
    score = float(inl.sum()) * min(collinear, 50.0) / (1.0 + (rmse if rmse is not None else 4.0))
    return {'k': k, 'b': b, 'polarity': None, 'score': score,
            'points': P, 'inliers': inl, 'segments': None,
            'evidence': {'row': row, 'edges': edges, 'eig_ratio': collinear}}


def _run_mode(gray: np.ndarray, mode: str, max_slope: float, band: int,
              seed: int) -> Optional[Dict]:
    if mode == 'sobel_ransac':
        return _mode_sobel_ransac(gray, max_slope, seed)
    if mode == 'hough':
        return _mode_hough(gray, max_slope)
    if mode == 'otsu_column':
        return _mode_otsu_column(gray, max_slope, seed)
    if mode == 'fitline':
        return _mode_fitline(gray, max_slope, band)
    if mode == 'edge_ransac':
        return _mode_edge_ransac(gray, max_slope, band, seed)
    if mode == 'pca':
        return _mode_pca(gray, max_slope, band)
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# 主流程：基线识别
# --------------------------------------------------------------------------- #
def detect_baseline(bgr: np.ndarray,
                    loc: Optional[Dict] = None,
                    *,
                    mode: str = 'sobel_ransac',
                    max_slope: float = 0.2,
                    band: int = 60,
                    seed: int = 0) -> Dict:
    """在液滴图像中识别台面基线。

    参数：
        bgr        —— 图像
        loc        —— Pre-process.locate_drop 的结果（可选，用于极性参考与偏差对比）
        mode       —— 算法模式，见 BASELINE_MODES
        max_slope  —— 允许的最大 |斜率|（超过则判定失败）
        band       —— 拟合类算法（fitline/edge_ransac/pca）使用的粗定位带宽
        seed       —— RANSAC 随机种子

    返回 dict：
        ok, mode, mode_label, k, b, invert, polarity, slope_deg, score,
        rmse, quality, n_points, n_inliers, endpoints, reference,
        evidence_vis, profile_vis, vis（失败时还有 error）
    """
    if mode not in BASELINE_MODES:
        raise ValueError(f'未知模式 {mode!r}，可选：{list(BASELINE_MODES)}')

    gray = _to_gray(bgr)
    h, w = gray.shape
    if h < 8 or w < 8:
        return {'ok': False, 'mode': mode, 'mode_label': BASELINE_MODES[mode],
                'error': '图像过小'}

    cand = _run_mode(gray, mode, max_slope, band, seed)
    if cand is None:
        return {'ok': False, 'mode': mode, 'mode_label': BASELINE_MODES[mode],
                'error': '未识别到基线（该算法未产生候选）'}

    k = float(cand['k'])
    b = float(cand['b'])
    if abs(k) > max_slope:
        return {'ok': False, 'mode': mode, 'mode_label': BASELINE_MODES[mode],
                'error': f'识别到的直线斜率过大（k={k:.4f}）', 'k': k, 'b': b}

    # 直线应穿过图像附近区域
    y_left = b
    y_right = k * (w - 1) + b
    if max(y_left, y_right) < -h or min(y_left, y_right) > 2 * h:
        return {'ok': False, 'mode': mode, 'mode_label': BASELINE_MODES[mode],
                'error': '识别到的直线未穿过图像区域', 'k': k, 'b': b}

    # 极性 / 液滴侧
    if cand.get('polarity') in (1.0, -1.0):
        polarity = float(cand['polarity'])
    elif loc and loc.get('ok'):
        polarity = -1.0 if loc.get('invert') else 1.0
    else:
        polarity = _line_polarity(_sobel_y(gray), k, b, w, h)
    invert = bool(polarity < 0)

    # 指标
    pts = cand.get('points')
    inl = cand.get('inliers')
    rmse = None
    if pts is not None and inl is not None and len(inl) == len(pts) and inl.any():
        rmse = _rmse(pts[inl], k, b)
    quality = float(np.clip(1.0 / (1.0 + (rmse if rmse is not None else 4.0)),
                            0.0, 1.0))

    # 参考（Pre-process 内置基线）与偏差
    reference = None
    if loc and loc.get('ok'):
        ref_k = float(loc['k'])
        ref_b = float(loc['b'])
        y_self = k * (w / 2.0) + b
        y_ref = ref_k * (w / 2.0) + ref_b
        reference = {
            'k': ref_k, 'b': ref_b,
            'invert': bool(loc.get('invert', False)),
            'angle_diff_deg': abs(math.degrees(math.atan(k))
                                  - math.degrees(math.atan(ref_k))),
            'offset_px': abs(y_self - y_ref),
        }

    evidence_vis = _evidence_vis(bgr, cand, mode, k, b)
    profile_vis = _profile_vis(cand)
    vis = annotate_baseline(bgr, k, b, cand, mode, invert, rmse)

    return {
        'ok': True,
        'mode': mode,
        'mode_label': BASELINE_MODES[mode],
        'k': k,
        'b': b,
        'invert': invert,
        'polarity': polarity,
        'slope_deg': math.degrees(math.atan(k)),
        'score': float(cand['score']),
        'rmse': rmse,
        'quality': quality,
        'n_points': int(len(pts)) if pts is not None else 0,
        'n_inliers': int(inl.sum()) if inl is not None else 0,
        'endpoints': _line_endpoints(k, b, w, h),
        'reference': reference,
        'evidence_vis': evidence_vis,
        'profile_vis': profile_vis,
        'vis': vis,
    }


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def _evidence_vis(bgr: np.ndarray, cand: Dict, mode: str, k: float,
                  b: float) -> np.ndarray:
    """算法证据图：候选点（绿=内点，蓝=外点）或 Hough 候选线段。"""
    vis = bgr.copy()
    h, w = vis.shape[:2]

    for seg in (cand.get('segments') or []):
        x1, y1, x2, y2 = [int(round(v)) for v in seg]
        cv2.line(vis, (x1, y1), (x2, y2), (0, 220, 255), 1, cv2.LINE_AA)

    pts = cand.get('points')
    if pts is not None and len(pts):
        inl = cand.get('inliers')
        if inl is not None and len(inl) == len(pts):
            for (x, y), ok in zip(pts, inl):
                color = (0, 230, 120) if ok else (90, 90, 220)
                cv2.circle(vis, (int(round(x)), int(round(y))), 2, color, -1)
        else:
            for x, y in pts:
                cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 220, 255), -1)

    p0, p1 = _line_endpoints(k, b, w, h)
    cv2.line(vis, (int(round(p0[0])), int(round(p0[1]))),
             (int(round(p1[0])), int(round(p1[1]))), (80, 200, 255), 2, cv2.LINE_AA)

    s = max(1.0, w / 1200.0)
    cv2.putText(vis, f'evidence · {BASELINE_MODES.get(mode, mode)}',
                (int(14 * s), int(32 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s,
                (245, 245, 245), max(1, int(1.6 * s)), cv2.LINE_AA)
    return vis


def _profile_vis(cand: Dict) -> Optional[np.ndarray]:
    """行剖面曲线图（供基于行剖面的模式展示内部证据）。"""
    prof = (cand.get('evidence') or {}).get('profile')
    if prof is None:
        return None
    prof = np.asarray(prof, dtype=np.float64).ravel()
    if prof.size < 2:
        return None

    W, H = 640, 240
    canvas = np.full((H, W, 3), (24, 27, 32), np.uint8)
    for gy in range(0, H, 40):
        cv2.line(canvas, (0, gy), (W - 1, gy), (40, 44, 50), 1)

    lo, hi = float(prof.min()), float(prof.max())
    rng = (hi - lo) or 1.0
    xs = np.linspace(0, W - 1, prof.size)
    ys = H - 10 - (prof - lo) / rng * (H - 24)
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, (80, 170, 255), 2, cv2.LINE_AA)

    peak = int(np.argmax(prof))
    px = int(round((peak / max(1, prof.size - 1)) * (W - 1)))
    py = int(round(ys[peak]))
    cv2.drawMarker(canvas, (px, py), (0, 220, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(canvas, f'row profile · peak y={peak}', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 230, 240), 1, cv2.LINE_AA)
    return canvas


def annotate_baseline(bgr: np.ndarray, k: float, b: float, cand: Dict,
                      mode: str, invert: bool,
                      rmse: Optional[float]) -> np.ndarray:
    """最终结果叠加图：内点 + 基线 + 液滴侧箭头 + 参数文字。"""
    vis = bgr.copy()
    h, w = vis.shape[:2]

    pts = cand.get('points')
    inl = cand.get('inliers')
    if pts is not None and len(pts) and inl is not None and len(inl) == len(pts):
        for (x, y), ok in zip(pts, inl):
            if ok:
                cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 230, 120), -1)

    p0, p1 = _line_endpoints(k, b, w, h)
    cv2.line(vis, (int(round(p0[0])), int(round(p0[1]))),
             (int(round(p1[0])), int(round(p1[1]))), (80, 80, 255), 3, cv2.LINE_AA)

    # 液滴侧箭头（正立：基线下方是衬底，液滴在上）
    cx = w // 2
    cy = int(round(k * cx + b))
    tip = cy - 26 if not invert else cy + 26
    tip = max(8, min(h - 8, tip))
    start = cy - 10 if not invert else cy + 10
    start = max(0, min(h - 1, start))
    cv2.arrowedLine(vis, (cx, start), (cx, tip), (80, 220, 255), 2,
                    cv2.LINE_AA, tipLength=0.5)

    s = max(1.0, w / 1200.0)
    txt = [
        f'baseline [{BASELINE_MODES.get(mode, mode)}]',
        f'y = {k:+.4f} x + {b:.1f}   slope {math.degrees(math.atan(k)):+.2f} deg',
        'side: ' + ('droplet below (inverted)' if invert else 'droplet above'),
    ]
    if rmse is not None:
        n_in = int(inl.sum()) if inl is not None else 0
        n_pts = len(pts) if pts is not None else 0
        txt.append(f'RMSE {rmse:.2f} px · inliers {n_in}/{n_pts}')
    yy = int(34 * s)
    for t in txt:
        cv2.putText(vis, t, (int(14 * s), yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s,
                    (245, 245, 245), max(1, int(1.7 * s)), cv2.LINE_AA)
        yy += int(28 * s)
    return vis


# --------------------------------------------------------------------------- #
# 供 UI 使用的完整流水线：预处理（参考）→ 基线识别
# --------------------------------------------------------------------------- #
def process_baseline(bgr: np.ndarray, *,
                     use_sr: bool = False,
                     sr_scale: float = 3.0,
                     mode: str = 'sobel_ransac',
                     max_slope: float = 0.2,
                     band: int = 60) -> Dict:
    """预处理（超分可选 + 液滴定位参考）+ 基线识别流水线，供 UI 展示。"""
    try:
        import preprocess as pp  # type: ignore
    except ImportError:
        import os as _os
        import sys as _sys
        parent = _os.path.abspath(_os.path.join(_os.path.dirname(__file__),
                                                '..', 'Pre-process'))
        if parent not in _sys.path:
            _sys.path.insert(0, parent)
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
    stages.append(('液滴定位（预处理参考）', pp.annotate(work, loc),
                   str(loc['bbox']) if loc.get('ok') else '定位失败'))

    result = detect_baseline(work, loc, mode=mode, max_slope=max_slope, band=band)

    if result.get('ok'):
        if result.get('profile_vis') is not None:
            stages.append(('证据 · 行剖面', result['profile_vis'],
                           '算法内部使用的水平边缘强度剖面'))
        stages.append((f'证据 · {result["mode_label"]}', result['evidence_vis'],
                       '候选点 / 候选线段（绿=内点）'))
        stages.append(('基线识别结果', result['vis'],
                       f'y = {result["k"]:+.4f}x + {result["b"]:.1f}'))
    else:
        stages.append(('基线识别失败', work.copy(), result.get('error', '')))

    return {'stages': stages, 'result': result, 'sr_active': sr_ok,
            'ok': result.get('ok', False)}
