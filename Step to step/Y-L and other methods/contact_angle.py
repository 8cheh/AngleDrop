"""Y-L and other methods · 接触角拟合核心代码（多种拟合方式）。

本目录是 AngleDrop「Step to step」的第 4 步 —— 接触角拟合：
在第 1 步 Pre-process（掩码 + 基线）、第 2 步 Sobel-lemma（接触点）、
第 3 步 baseline-lemma（基线细化）的输出之上，用**可切换的拟合模型**
从液滴轮廓求左右接触角。

设计原则（独立实现，不复制 Opendrop / cadrop 源码）：

    1. 统一坐标系：把图像点变换到「基线坐标系 (r, z)」，
       r 沿基线向右，z 垂直基线指向液滴内部（z>0 即液相一侧）；
    2. 接触角符号约定：令接触点处切线向上为 (t_r, t_z)，则
           θ_left  = atan2(t_z,  t_r)
           θ_right = atan2(t_z, -t_r)
       全程不取绝对值，因此疏水液滴（θ>90°）也能正确表示；
    3. 几何模型在**接触点附近的局部窗口**内拟合（OpenDrop 的局部拟合思想），
       物理模型 Young–Laplace 则对**整条单侧轮廓**做全局拟合（ADSA）；
    4. 每种方法独立给出「接触角 + 残差 RMS + 拟合曲线」，便于横向对比。

可切换模式（ANGLE_METHODS）：

    young_laplace    Young–Laplace 双参数（顶点曲率 R0 + 毛细常数 β，含重力项）
    young_laplace_cap Young–Laplace 单参数（β=0，退化为球冠）
    circle           圆拟合（Kasa 代数初值 + 几何距离 Gauss–Newton 精修）
    ellipse          椭圆（Halir–Flusser 直接最小二乘拟合）
    poly2            二次多项式切线 r = f(z)
    poly3            三次多项式切线
    spline           三次平滑样条切线
    line             直线（楔形）拟合
    auto             自适应：跑所有方法，按残差择优

只做核心拟合，不负责 Web 接口（app.py 负责）；所有函数都可单独调用。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # scipy 用于 Y-L 数值积分与优化；缺失时自动降级（Y-L 不可用）
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares
    from scipy.interpolate import UnivariateSpline
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - 仅在缺少 scipy 时触发
    _HAVE_SCIPY = False

__all__ = [
    'ANGLE_METHODS',
    'angle_from_tangent',
    'fit_circle',
    'fit_ellipse',
    'fit_poly',
    'fit_line',
    'young_laplace_profile',
    'detect_contact_angle',
    'annotate_contact_angle',
    'process_contact_angle',
]

ANGLE_METHODS = {
    'young_laplace': 'Young–Laplace 双参数（含重力项 / ADSA）',
    'young_laplace_cap': 'Young–Laplace 单参数（球冠极限）',
    'circle': '圆拟合（Kasa + 几何精修）',
    'ellipse': '椭圆（Halir–Flusser 直接拟合）',
    'poly2': '二次多项式切线',
    'poly3': '三次多项式切线',
    'spline': '三次平滑样条切线',
    'line': '直线（楔形）拟合',
    'auto': '自适应（按残差择优）',
}

# auto 模式下参与竞争的「实际模型」（不含 auto 自身）
_REAL_METHODS = [m for m in ANGLE_METHODS if m != 'auto']


# --------------------------------------------------------------------------- #
# 基线坐标系变换
# --------------------------------------------------------------------------- #
def _line_frame(k: float, b: float, w: int, invert: bool):
    """返回基线坐标系的 (origin, unit, perp)。

    origin 为直线与图像左边界交点 (0, b)；
    unit 为沿基线向右的单位向量；perp 为垂直基线的单位向量，
    invert=False 时指向图像上方（z>0 为液滴侧），invert=True 时指向下方。
    """
    p0 = np.array([0.0, float(b)])
    p1 = np.array([float(w - 1), k * (w - 1) + b])
    d = p1 - p0
    n = math.hypot(float(d[0]), float(d[1]))
    if n < 1e-9:
        n = 1.0
    unit = d / n
    perp = np.array([unit[1], -unit[0]])
    if invert:
        perp = -perp
    return p0, unit, perp


def _xy_to_rz(P: np.ndarray, origin, unit, perp) -> np.ndarray:
    """图像坐标 (N,2) -> 基线坐标 (N,2) = (r, z)。"""
    P = np.asarray(P, dtype=np.float64)
    rel = P - origin[None, :]
    return np.stack([rel @ unit, rel @ perp], axis=1)


def _rz_to_xy(rz: np.ndarray, origin, unit, perp) -> np.ndarray:
    """基线坐标 (N,2) -> 图像坐标 (N,2)。"""
    rz = np.asarray(rz, dtype=np.float64)
    return origin[None, :] + rz[:, 0:1] * unit[None, :] + rz[:, 1:2] * perp[None, :]


# --------------------------------------------------------------------------- #
# 轮廓提取：逐行极值 + 法向亚像素细化
# --------------------------------------------------------------------------- #
def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def profile_points(gray: np.ndarray, mask: np.ndarray,
                   grad_frac: float = 0.18, grad_min: float = 6.0) -> np.ndarray:
    """从掩码提取液滴轮廓点；逐行取左右极值，再沿边缘法向亚像素细化。

    逐行极值天然排除了掩码在台面上被平切后内部的大片像素，
    避免这些假边缘把接触点附近的拟合拉直。返回 (N,2) 图像坐标。

    梯度过滤是关键：掩码在基线上方被平切，切口位于均匀亮区，
    不剔除会把近接触区的拟合拉向竖直。
    """
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    order = np.argsort(ys, kind='stable')
    ys, xs = ys[order], xs[order]
    bounds = np.searchsorted(ys, np.unique(ys), side='left')
    bounds = np.append(bounds, ys.size)
    pts: List[Tuple[float, float]] = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        row = xs[s:e]
        y = ys[s]
        pts.append((float(row.min()), float(y)))
        if row.max() != row.min():
            pts.append((float(row.max()), float(y)))
    P = np.array(pts, dtype=np.float64)
    if P.shape[0] < 8:
        return P

    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    x, y = P[:, 0], P[:, 1]
    m0 = _bilinear(mag, x, y)
    keep = m0 > max(grad_min, grad_frac * float(np.percentile(m0, 90)))
    if keep.sum() >= 20:
        P = P[keep]
        x, y, m0 = P[:, 0], P[:, 1], m0[keep]

    nx, ny = _bilinear(gx, x, y), _bilinear(gy, x, y)
    nn = np.hypot(nx, ny) + 1e-9
    nx, ny = nx / nn, ny / nn
    mm = _bilinear(mag, x - nx, y - ny)
    mp = _bilinear(mag, x + nx, y + ny)
    den = mm - 2.0 * m0 + mp
    t = np.where(np.abs(den) > 1e-6,
                 0.5 * (mm - mp) / np.where(np.abs(den) > 1e-6, den, 1.0), 0.0)
    t = np.clip(t, -1.0, 1.0)
    return np.stack([x + t * nx, y + t * ny], axis=1)


# --------------------------------------------------------------------------- #
# 接触角：由切线向量得到角度（无绝对值，支持 θ>90°）
# --------------------------------------------------------------------------- #
def angle_from_tangent(tr: float, tz: float, side: str) -> Optional[float]:
    """由轮廓切线向量 (t_r, t_z) 求接触角（度）。

    切线统一朝上（t_z>0）；side='left' 时固体方向指向 +r，
    side='right' 时指向 -r，因此疏水角可自然落到 (90°, 180°)。
    """
    if not (np.isfinite(tr) and np.isfinite(tz)):
        return None
    if tz < 0:
        tr, tz = -tr, -tz
    if tz <= 1e-12:
        return None
    a = math.atan2(tz, tr if side == 'left' else -tr)
    return math.degrees(a)


# --------------------------------------------------------------------------- #
# 几何模型 1：圆（Kasa 代数初值 + 几何距离 Gauss–Newton）
# --------------------------------------------------------------------------- #
def fit_circle(r: np.ndarray, z: np.ndarray):
    """最小二乘圆拟合，返回 (rc, zc, R, rms)；失败返回 None。"""
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if r.size < 4:
        return None
    A = np.column_stack([r, z, np.ones_like(r)])
    rhs = r ** 2 + z ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    rc, zc = sol[0] / 2.0, sol[1] / 2.0
    R2 = sol[2] + rc ** 2 + zc ** 2
    if not np.isfinite(R2) or R2 <= 0:
        return None
    R = math.sqrt(R2)

    for _ in range(30):
        d = np.hypot(r - rc, z - zc)
        if np.any(d < 1e-12):
            break
        J = np.column_stack([-(r - rc) / d, -(z - zc) / d, -np.ones_like(d)])
        try:
            upd, *_ = np.linalg.lstsq(J, -(d - R), rcond=None)
        except np.linalg.LinAlgError:
            break
        rc += upd[0]
        zc += upd[1]
        R += upd[2]
        if np.linalg.norm(upd) < 1e-10:
            break
    if not (np.isfinite(rc) and np.isfinite(zc) and np.isfinite(R)) or R <= 0:
        return None
    rms = float(np.sqrt(np.mean((np.hypot(r - rc, z - zc) - R) ** 2)))
    return float(rc), float(zc), float(R), rms


def circle_contact(rc: float, zc: float, R: float, side: str,
                   near_r: Optional[float] = None):
    """圆与基线 z=0 的交点，以及该处接触角。"""
    if abs(zc) >= R:
        return None, None
    off = math.sqrt(R * R - zc * zc)
    cands = (rc - off, rc + off)
    if near_r is not None:
        r0 = min(cands, key=lambda c: abs(c - near_r))
    else:
        r0 = cands[0] if side == 'left' else cands[1]
    # 接触点处半径方向 = (r0-rc, -zc)，切线与其正交 → (zc, r0-rc)
    tr, tz = zc, (r0 - rc)
    return angle_from_tangent(tr, tz, side), float(r0)


# --------------------------------------------------------------------------- #
# 几何模型 2：椭圆 / 一般二次曲线（Halir–Flusser 直接拟合）
# --------------------------------------------------------------------------- #
def fit_ellipse(r: np.ndarray, z: np.ndarray) -> Optional[np.ndarray]:
    """直接最小二乘椭圆拟合，返回二次曲线系数 (A,B,C,D,E,F)。"""
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if r.size < 6:
        return None
    D1 = np.column_stack([r * r, r * z, z * z])
    D2 = np.column_stack([r, z, np.ones_like(r)])
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    M = np.vstack([M[2, :] / 2.0, -M[1, :], M[0, :] / 2.0])
    try:
        ev, evec = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    cond = 4.0 * evec[0, :] * evec[2, :] - evec[1, :] ** 2
    ok = np.where((cond > 0) & np.isfinite(ev))[0]
    if ok.size == 0:
        return None
    a1 = np.real(evec[:, ok[0]])
    co = np.concatenate([a1, T @ a1])
    return co if np.all(np.isfinite(co)) else None


def conic_rms(co: np.ndarray, r: np.ndarray, z: np.ndarray) -> float:
    A, B, C, D, E, F = co
    v = A * r * r + B * r * z + C * z * z + D * r + E * z + F
    gr = 2 * A * r + B * z + D
    gz = B * r + 2 * C * z + E
    g = np.hypot(gr, gz) + 1e-12
    return float(np.sqrt(np.mean((v / g) ** 2)))


def conic_contact(co: np.ndarray, side: str, near_r: Optional[float] = None):
    """二次曲线与基线 z=0 的交点，以及该处接触角。"""
    A, B, C, D, E, F = co
    if abs(A) < 1e-15:
        return None, None
    disc = D * D - 4.0 * A * F
    if disc < 0:
        return None, None
    s = math.sqrt(disc)
    cands = ((-D - s) / (2 * A), (-D + s) / (2 * A))
    if near_r is not None:
        r0 = min(cands, key=lambda c: abs(c - near_r))
    else:
        r0 = min(cands) if side == 'left' else max(cands)
    Fr = 2 * A * r0 + D          # ∂F/∂r
    Fz = B * r0 + E              # ∂F/∂z
    # 切线垂直于曲线梯度 → (t_r, t_z) ∝ (-Fz, Fr)
    return angle_from_tangent(-Fz, Fr, side), float(r0)


# --------------------------------------------------------------------------- #
# 几何模型 3：多项式 / 样条（r = f(z)）
# --------------------------------------------------------------------------- #
def fit_poly(r: np.ndarray, z: np.ndarray, deg: int = 2):
    """r = f(z) 多项式拟合，返回 (coeffs, r0, dr/dz|0, rms)。"""
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if r.size < deg + 2:
        return None
    try:
        co = np.polyfit(z, r, deg)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(co)):
        return None
    d = np.polyder(co)
    rms = float(np.sqrt(np.mean((np.polyval(co, z) - r) ** 2)))
    return co, float(np.polyval(co, 0.0)), float(np.polyval(d, 0.0)), rms


def fit_line(r: np.ndarray, z: np.ndarray):
    """r = m z + c 的总体最小二乘，返回 (m, c, rms, max_abs)。"""
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if r.size < 3:
        return None
    try:
        m, c = np.polyfit(z, r, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not (np.isfinite(m) and np.isfinite(c)):
        return None
    res = np.polyval([m, c], z) - r
    return float(m), float(c), float(np.sqrt(np.mean(res ** 2))), float(np.abs(res).max())


def fit_spline(r: np.ndarray, z: np.ndarray):
    """三次平滑样条 r = f(z)，返回 (spline, r0, dr/dz|0, rms)。"""
    if not _HAVE_SCIPY:
        return None
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if r.size < 6:
        return None
    order = np.argsort(z)
    zs, rs = z[order], r[order]
    # z 去重（同深度只保留均值），避免样条节点重复
    uz, inv = np.unique(np.round(zs, 6), return_inverse=True)
    if uz.size < 6:
        return None
    means = np.bincount(inv, weights=rs) / np.bincount(inv)
    try:
        spl = UnivariateSpline(uz, means, k=min(3, uz.size - 1),
                               s=float(uz.size) * 0.5 ** 2)
        r0 = float(spl(0.0))
        slope = float(spl.derivative()(0.0))
        if not (np.isfinite(r0) and np.isfinite(slope)):
            return None
        rms = float(np.sqrt(np.mean((spl(zs) - rs) ** 2)))
    except Exception:
        return None
    return spl, r0, slope, rms


# --------------------------------------------------------------------------- #
# 物理模型：轴对称 Young–Laplace 方程（ADSA）
# --------------------------------------------------------------------------- #
def young_laplace_profile(b: float, lam: float, d_max: float = 1.0,
                          n: int = 600, s_max: float = 4000.0,
                          phi_max: float = math.radians(185.0)):
    """数值积分无量纲轴对称 Young–Laplace 方程。

    以顶点为原点、z 沿重力方向向下（D 为顶点以下的深度），弧长 S 为自变量：

        dX/dS = cos(phi)
        dD/dS = sin(phi)
        dphi/dS = 2*b + lam*D - sin(phi)/X

    其中 b = L / R0（L 为长度尺度，R0 为顶点曲率半径），
    lam = rho*g*L^2/gamma 为无量纲毛细常数（Bond 数），重力项 +lam*D
    使液滴随深度增加而逐渐摊平（正立液滴）。

    返回 (X, D, phi) 三条数组；若积分未到达 D=d_max 则返回 None。
    """
    if not _HAVE_SCIPY:
        return None
    if not (np.isfinite(b) and np.isfinite(lam)) or b <= 0:
        return None
    s0 = 1e-6
    y0 = [s0, 0.0, b * s0]

    def rhs(_s, y):
        X, D, phi = y
        if X < 1e-8:
            dphi = b                    # 顶点极限：2b - sin(phi)/X -> b
        else:
            dphi = 2.0 * b + lam * D - math.sin(phi) / X
        return [math.cos(phi), math.sin(phi), dphi]

    def ev_d(_s, y):
        return y[1] - d_max
    ev_d.terminal = True
    ev_d.direction = 1

    def ev_x(_s, y):
        return y[0] - 30.0
    ev_x.terminal = True
    ev_x.direction = 1

    def ev_phi(_s, y):
        return y[2] - phi_max
    ev_phi.terminal = True
    ev_phi.direction = 1

    try:
        # 关键：同时监测 D=d_max、X 发散、phi 超过 ~185°（防止轮廓自卷成螺旋）
        sol = solve_ivp(rhs, [s0, s_max], y0, method='RK45',
                        events=[ev_d, ev_x, ev_phi], dense_output=True,
                        rtol=1e-6, atol=1e-9, max_step=0.25)
    except Exception:
        return None
    if (not sol.success) or sol.t.size < 3:
        return None
    if sol.y[1, -1] < d_max - 1e-3:
        return None
    if sol.y[2, -1] > phi_max + 1e-6:
        return None
    S = np.linspace(s0, float(sol.t[-1]), n)
    Y = sol.sol(S)
    return Y[0], Y[1], Y[2]


def _yl_fit(x_meas: np.ndarray, d_meas: np.ndarray, cap: bool,
            b_init: float) -> Optional[Dict]:
    """把 Y-L 理论曲线拟合到实测单侧轮廓（最近点距离，多初值防局部极小）。"""
    if not _HAVE_SCIPY or x_meas.size < 8:
        return None

    def make_residuals(cap_flag: bool):
        def residuals(params):
            b = max(float(params[0]), 1e-6)
            lam = 0.0 if cap_flag else max(float(params[1]), 0.0)
            prof = young_laplace_profile(b, lam, d_max=1.0, n=300)
            if prof is None:
                return np.full(x_meas.size, 1e3)
            X, D, _phi = prof
            d2 = (x_meas[:, None] - X[None, :]) ** 2 + (d_meas[:, None] - D[None, :]) ** 2
            return np.sqrt(d2.min(axis=1))
        return residuals

    b_lo, b_hi = 2e-4, 20.0
    b_init = float(np.clip(b_init, b_lo * 1.5, b_hi * 0.5))
    if cap:
        starts = [[b_init]]
    else:
        # 多初值：从「球冠解」附近出发，毛细常数由小到大试探
        starts = [[b_init, 0.0], [b_init, 0.3], [b_init * 1.3, 1.0],
                  [b_init * 0.7, 0.1]]
    best = None
    for x0 in starts:
        lo = [b_lo, 0.0][:len(x0)]
        hi = [b_hi, 20.0][:len(x0)]
        try:
            res = least_squares(make_residuals(cap), x0, bounds=(lo, hi),
                                method='trf', loss='soft_l1', f_scale=0.02,
                                x_scale='jac', max_nfev=300)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return None
    b = max(float(best.x[0]), 1e-6)
    lam = 0.0 if cap else max(float(best.x[1]), 0.0)
    prof = young_laplace_profile(b, lam, d_max=1.0, n=2000)
    if prof is None:
        return None
    X, D, phi = prof
    i = int(np.argmin(np.abs(D - 1.0)))
    ang = math.degrees(float(phi[i]))
    if not (0.0 < ang < 180.0):
        return None
    # 真实点-曲线最近距离的 RMS（least_squares 的 fun 是 soft_l1 变换后的值，
    # 不能直接当残差报告）
    d2 = (x_meas[:, None] - X[None, :]) ** 2 + (d_meas[:, None] - D[None, :]) ** 2
    rms_norm = float(np.sqrt(np.mean(d2.min(axis=1))))
    return {'b': b, 'lam': lam, 'angle_deg': ang, 'rms_norm': rms_norm,
            'x_at_base': float(X[i]), 'X': X, 'D': D, 'phi': phi,
            'cost': float(best.cost)}


# --------------------------------------------------------------------------- #
# 单侧拟合调度
# --------------------------------------------------------------------------- #
def _fit_side(r: np.ndarray, z: np.ndarray, contact_r: float, side: str,
              method: str, *, base_w: float, height: float,
              r_apex: float) -> Dict:
    """在单侧轮廓点上按 `method` 拟合，返回结果 dict。"""
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    out: Dict = {'ok': False, 'method': method, 'angle_deg': None,
                 'rms_px': None, 'n_points': int(r.size), 'r0': None,
                 'model': None, 'z_hi': float(z.max()) if z.size else 0.0}
    if r.size < 5:
        return out

    if method in ('young_laplace', 'young_laplace_cap'):
        cap = method == 'young_laplace_cap'
        H = float(height) if height > 1e-6 else float(z.max())
        H = max(H, 1e-6)
        x_meas = np.abs(r - r_apex) / H
        d_meas = np.clip((H - z) / H, 0.0, 1.0)
        x_c = abs(contact_r - r_apex) / H
        # 加入接触点锚定（z=0 → D=1）
        x_meas = np.concatenate([x_meas, [x_c]])
        d_meas = np.concatenate([d_meas, [1.0]])
        # 初始 b：用圆拟合估计接触角，再由球冠关系 b ~ 1-cos(theta)
        b_init = 0.5
        cf = fit_circle(r, z)
        if cf is not None:
            _rc, _zc, _R, _ = cf
            ang0, _r0 = circle_contact(_rc, _zc, _R, side, near_r=contact_r)
            if ang0 is not None:
                b_init = max(2e-4, 1.0 - math.cos(math.radians(ang0)))
        yl = _yl_fit(x_meas, d_meas, cap, b_init)
        if yl is None:
            return out
        out.update(ok=True, angle_deg=yl['angle_deg'],
                   rms_px=yl['rms_norm'] * H, r0=yl['x_at_base'] * H,
                   model={'kind': 'yl', 'cap': cap, 'b': yl['b'],
                          'lam': yl['lam'], 'r_apex': r_apex, 'H': H,
                          'X': yl['X'], 'D': yl['D'], 'phi': yl['phi'],
                          'contact_r': contact_r})
        return out

    # 几何模型：先把接触点的 (r=contact_r, z=0) 也纳入以锚定外推
    if method in ('poly2', 'poly3'):
        deg = 2 if method == 'poly2' else 3
        f = fit_poly(r, z, deg)
        if f is None:
            return out
        co, r0, slope, rms = f
        ang = angle_from_tangent(slope, 1.0, side)
        if ang is None:
            return out
        out.update(ok=True, angle_deg=ang, rms_px=rms, r0=r0, model={'kind': 'poly', 'co': co})
    elif method == 'spline':
        f = fit_spline(r, z)
        if f is None:
            return out
        spl, r0, slope, rms = f
        ang = angle_from_tangent(slope, 1.0, side)
        if ang is None:
            return out
        out.update(ok=True, angle_deg=ang, rms_px=rms, r0=r0, model={'kind': 'spline', 'spl': spl})
    elif method == 'line':
        f = fit_line(r, z)
        if f is None:
            return out
        m, c, rms, _mx = f
        ang = angle_from_tangent(m, 1.0, side)
        if ang is None:
            return out
        out.update(ok=True, angle_deg=ang, rms_px=rms, r0=c, model={'kind': 'line', 'm': m, 'c': c})
    elif method == 'circle':
        f = fit_circle(r, z)
        if f is None:
            return out
        rc, zc, R, rms = f
        ang, r0 = circle_contact(rc, zc, R, side, near_r=contact_r)
        if ang is None:
            return out
        out.update(ok=True, angle_deg=ang, rms_px=rms, r0=r0,
                   model={'kind': 'circle', 'rc': rc, 'zc': zc, 'R': R, 'r0': r0})
    elif method == 'ellipse':
        co = fit_ellipse(r, z)
        if co is None:
            return out
        ang, r0 = conic_contact(co, side, near_r=contact_r)
        if ang is None:
            return out
        out.update(ok=True, angle_deg=ang, rms_px=conic_rms(co, r, z), r0=r0,
                   model={'kind': 'conic', 'co': co, 'r0': r0})
    else:
        return out
    return out


def _sample_curve(model: Dict, side: str, z_hi: float, n: int = 80):
    """把拟合模型采样成基线坐标下的 (r, z) 曲线，用于可视化。"""
    if model is None:
        return None
    kind = model['kind']
    if kind == 'yl':
        X, D = model['X'], model['D']
        H, r_apex = model['H'], model['r_apex']
        sign = -1.0 if side == 'left' else 1.0
        r = r_apex + sign * X * H
        z = H * (1.0 - D)
        return np.stack([r, z], axis=1)
    if kind == 'poly':
        zs = np.linspace(0.0, max(z_hi, 1e-6), n)
        return np.stack([np.polyval(model['co'], zs), zs], axis=1)
    if kind == 'spline':
        zs = np.linspace(0.0, max(z_hi, 1e-6), n)
        try:
            rs = model['spl'](zs)
        except Exception:
            return None
        return np.stack([rs, zs], axis=1)
    if kind == 'line':
        zs = np.linspace(0.0, max(z_hi, 1e-6), n)
        return np.stack([model['m'] * zs + model['c'], zs], axis=1)
    if kind == 'circle':
        rc, zc, R = model['rc'], model['zc'], model['R']
        r0 = model['r0']
        if abs(zc) >= R:
            return None
        phi0 = math.atan2(0.0 - zc, r0 - rc)
        z_top = min(z_hi, zc + R)
        val = max(-1.0, min(1.0, (z_top - zc) / R))
        phi1 = math.asin(val)
        if abs(phi1 - phi0) > math.pi / 2:
            phi1 = math.pi - phi1 if phi1 >= 0 else -math.pi - phi1
        phis = np.linspace(phi0, phi1, n)
        return np.stack([rc + R * np.cos(phis), zc + R * np.sin(phis)], axis=1)
    if kind == 'conic':
        co = model['co']
        r0 = model['r0']
        A, B, C, D, E, F = co
        zs = np.linspace(0.0, max(z_hi, 1e-6), n)
        aa = A
        bb = B * zs + D
        cc = C * zs * zs + E * zs + F
        disc = bb * bb - 4 * aa * cc
        good = disc >= 0
        if good.sum() < 2:
            return None
        sq = np.sqrt(np.maximum(disc, 0))
        r1 = (-bb - sq) / (2 * aa)
        r2 = (-bb + sq) / (2 * aa)
        pick = np.where(np.abs(r1 - r0) < np.abs(r2 - r0), r1, r2)
        return np.stack([pick[good], zs[good]], axis=1)
    return None


# --------------------------------------------------------------------------- #
# 接触点（当上游未提供时由掩码 + 基线几何求交）
# --------------------------------------------------------------------------- #
def _zmap(k: float, b: float, w: int, h: int, invert: bool) -> np.ndarray:
    xx = np.arange(w, dtype=np.float64)[None, :]
    yy = np.arange(h, dtype=np.float64)[:, None]
    dx = w - 1.0
    dy = k * (w - 1.0)
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    px, py = uy, -ux
    if invert:
        px, py = -px, -py
    return (px * xx + py * (yy - b)).astype(np.float64)


def contacts_from_mask(mask: np.ndarray, k: float, b: float, invert: bool,
                       clearance: float = 3.0, band: float = 10.0):
    """由掩码外轮廓与基线的几何关系求左右接触点（基线坐标 r）。"""
    h, w = mask.shape
    zm = _zmap(k, b, w, h, invert)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    xs, ys = cnt[:, 0], cnt[:, 1]
    z = zm[np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)]
    near = (z > 0) & (z <= clearance + band)
    if not near.any():
        near = z > 0
    if not near.any():
        return None
    nxs, nys, nz = xs[near], ys[near], z[near]
    close = nz <= float(nz.min()) + 0.5
    cx, cy = nxs[close], nys[close]
    if cx.size < 2:
        return None
    li, ri = int(np.argmin(cx)), int(np.argmax(cx))
    if li == ri:
        return None
    return (float(cx[li]), float(cy[li])), (float(cx[ri]), float(cy[ri]))


def _preprocess_locate(bgr: np.ndarray) -> Dict:
    """懒加载 Pre-process 的 locate_drop，避免模块强耦合。"""
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


# --------------------------------------------------------------------------- #
# 主流程：接触角拟合
# --------------------------------------------------------------------------- #
def detect_contact_angle(bgr: np.ndarray,
                         loc: Optional[Dict] = None,
                         contact: Optional[Dict] = None,
                         *,
                         method: str = 'young_laplace',
                         window_frac: float = 0.25,
                         v_frac: float = 1.0,
                         min_points: int = 6,
                         compare: bool = False) -> Dict:
    """在液滴图像上拟合接触角。

    参数：
        bgr         —— 图像
        loc         —— Pre-process.locate_drop 的结果（含 mask / k / b / invert）
        contact     —— 第 2 步 detect_contact_line 的结果（含 left/right 的 x,y）；
                       为空时由掩码 + 基线内部求交
        method      —— 拟合方法，见 ANGLE_METHODS
        window_frac —— 几何模型局部拟合窗口（相对底宽）
        v_frac      —— 局部窗口的竖向上限（相对液滴高，默认 1.0=不限制）：
                       扁平液滴顶部平坦，可调小以只保留近接触区（需自行权衡）。
        compare     —— 是否额外跑所有方法，返回对比表

    返回 dict（成功时）：
        ok, method, method_label, window_frac,
        left/right: {ok, angle_deg, rms_px, n_points, r0, model},
        avg_angle, asymmetry, confidence,
        baseline: {k,b,invert}, base_width, height,
        comparison: [{method,label,ok,left,right,avg,rms_px}, ...],
        profile_rz, vis
    """
    if method not in ANGLE_METHODS:
        raise ValueError(f'未知方法 {method!r}，可选：{list(ANGLE_METHODS)}')
    if loc is None:
        loc = _preprocess_locate(bgr)
    if not loc.get('ok'):
        return {'ok': False, 'method': method, 'error': loc.get('error', '预处理未找到液滴')}

    mask = loc.get('mask')
    if mask is None or mask.shape[:2] != bgr.shape[:2]:
        return {'ok': False, 'method': method, 'error': '掩码尺寸与图像不一致'}

    k, b = float(loc['k']), float(loc['b'])
    invert = bool(loc.get('invert', False))
    h, w = mask.shape
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    origin, unit, perp = _line_frame(k, b, w, invert)

    # 1) 轮廓点 -> 基线坐标
    P = profile_points(gray, mask)
    if P.shape[0] < 12:
        return {'ok': False, 'method': method, 'error': '轮廓点太少'}
    rz = _xy_to_rz(P, origin, unit, perp)
    keep = rz[:, 1] > 0.5
    if keep.sum() < 12:
        keep = rz[:, 1] > 0.0
    P, rz = P[keep], rz[keep]
    r, z = rz[:, 0], rz[:, 1]

    # 2) 接触点
    if contact is not None and contact.get('left') and contact.get('right'):
        lp = np.array([contact['left']['x'], contact['left']['y']], dtype=np.float64)
        rp = np.array([contact['right']['x'], contact['right']['y']], dtype=np.float64)
        lrz = _xy_to_rz(lp[None, :], origin, unit, perp)[0]
        rrz = _xy_to_rz(rp[None, :], origin, unit, perp)[0]
        left_r, right_r = float(lrz[0]), float(rrz[0])
        left_xy = lp
        right_xy = rp
    else:
        cc = contacts_from_mask(mask, k, b, invert)
        if cc is None:
            return {'ok': False, 'method': method, 'error': '无法确定接触点'}
        (lx, ly), (rx, ry) = cc
        left_xy = np.array([lx, ly], dtype=np.float64)
        right_xy = np.array([rx, ry], dtype=np.float64)
        left_r = float(_xy_to_rz(left_xy[None, :], origin, unit, perp)[0, 0])
        right_r = float(_xy_to_rz(right_xy[None, :], origin, unit, perp)[0, 0])

    base_w = float(right_r - left_r)
    if base_w <= 5:
        return {'ok': False, 'method': method, 'error': f'底宽过小 ({base_w:.1f}px)'}
    height = float(z.max())
    # 疏水液滴顶部是「平顶」，单个最大 z 点可能落在平的边缘上；
    # 用顶部一小段内的左右极值中点作为对称轴的鲁棒估计。
    top_band = max(2.0, 0.03 * height)
    top = z > height - top_band
    if top.sum() >= 2:
        r_apex = float(0.5 * (r[top].min() + r[top].max()))
    else:
        r_apex = float(r[int(np.argmax(z))])

    # 3) 单侧窗口点
    def side_points(side: str, full: bool):
        sel = (r < r_apex) if side == 'left' else (r > r_apex)
        cr = left_r if side == 'left' else right_r
        if not full:
            d = np.hypot(r - cr, z)
            wsel = sel & (d < window_frac * base_w) & (z <= v_frac * height)
            if wsel.sum() >= max(min_points, 8):
                sel = wsel
            else:
                # 回退：取该侧离接触点最近的一批点（仍限制竖向范围）
                dd = np.where(sel & (z <= v_frac * height), d, np.inf)
                if not np.isfinite(dd).any():
                    dd = np.where(sel, d, np.inf)
                if np.isfinite(dd).any():
                    thr = np.percentile(dd[np.isfinite(dd)], 35)
                    sel = sel & (d <= thr)
        return sel

    def run_one(m: str) -> Dict:
        lsel = side_points('left', m in ('young_laplace', 'young_laplace_cap'))
        rsel = side_points('right', m in ('young_laplace', 'young_laplace_cap'))
        lf = _fit_side(r[lsel], z[lsel], left_r, 'left', m,
                       base_w=base_w, height=height, r_apex=r_apex
                       ) if lsel.sum() >= min_points else {
            'ok': False, 'method': m, 'angle_deg': None, 'rms_px': None,
            'n_points': int(lsel.sum()), 'r0': None, 'model': None, 'z_hi': height}
        rf = _fit_side(r[rsel], z[rsel], right_r, 'right', m,
                       base_w=base_w, height=height, r_apex=r_apex
                       ) if rsel.sum() >= min_points else {
            'ok': False, 'method': m, 'angle_deg': None, 'rms_px': None,
            'n_points': int(rsel.sum()), 'r0': None, 'model': None, 'z_hi': height}
        return {'left': lf, 'right': rf}

    # 4) 选方法
    fits: Dict[str, Dict] = {}
    if method == 'auto' or compare:
        for m in _REAL_METHODS:
            fits[m] = run_one(m)
    if method == 'auto':
        # 首选顺序：稳定 / 物理可解释的模型优先，避免样条/高次多项式过拟合。
        # 评分 = 相对残差 + 模型偏好惩罚 + 单侧失败惩罚 + 左右不对称惩罚。
        rank = {'circle': 0, 'young_laplace': 1, 'ellipse': 2, 'poly2': 3,
                'line': 4, 'young_laplace_cap': 5, 'poly3': 6, 'spline': 7}
        best_m, best_score = None, None
        for m, f in fits.items():
            ok_l, ok_r = f['left']['ok'], f['right']['ok']
            if not (ok_l or ok_r):
                continue
            rms_pair = [v for v in (f['left']['rms_px'], f['right']['rms_px'])
                        if v is not None]
            if not rms_pair:
                continue
            rms = float(np.mean(rms_pair))
            score = rms / max(base_w, 20.0) + 0.02 * rank.get(m, 8)
            if not (ok_l and ok_r):
                score += 0.05                       # 只有单侧成功 → 惩罚
            else:
                score += 0.002 * abs(f['left']['angle_deg'] - f['right']['angle_deg'])
            if best_score is None or score < best_score:
                best_score, best_m = score, m
        if best_m is None:
            return {'ok': False, 'method': 'auto', 'error': '所有拟合方法均失败'}
        chosen = best_m
        fits.setdefault(chosen, run_one(chosen))
    else:
        chosen = method
        fits[chosen] = run_one(chosen)

    lf, rf = fits[chosen]['left'], fits[chosen]['right']
    angles = [f['angle_deg'] for f in (lf, rf) if f['ok'] and f['angle_deg'] is not None]
    if not angles:
        return {'ok': False, 'method': chosen, 'method_label': ANGLE_METHODS[chosen],
                'error': '左右两侧拟合均失败'}

    avg = float(np.mean(angles))
    asym = abs(lf['angle_deg'] - rf['angle_deg']) if (lf['ok'] and rf['ok']) else None

    # 5) 置信度：残差相对底宽 + 点数 + 左右一致性
    conf = 1.0
    for f in (lf, rf):
        if f['ok']:
            conf *= math.exp(-(f['rms_px'] or 0.0) / max(0.02 * base_w, 2.0))
            conf *= min(1.0, f['n_points'] / 25.0)
        else:
            conf *= 0.35
    if asym is not None:
        conf *= math.exp(-asym / 25.0)
    conf = float(max(0.0, min(1.0, conf)))

    comparison = []
    if compare:
        for m, f in fits.items():
            lm, rm = f['left'], f['right']
            vals = [x for x in (lm['angle_deg'], rm['angle_deg']) if x is not None]
            rms_pair = [v for v in (lm['rms_px'], rm['rms_px']) if v is not None]
            comparison.append({
                'method': m, 'label': ANGLE_METHODS[m],
                'ok': bool(lm['ok'] or rm['ok']),
                'left': lm['angle_deg'], 'right': rm['angle_deg'],
                'avg': float(np.mean(vals)) if vals else None,
                'rms_px': float(np.mean(rms_pair)) if rms_pair else None,
                'n_left': lm['n_points'], 'n_right': rm['n_points'],
            })

    vis = annotate_contact_angle(bgr, chosen, lf, rf, left_xy, right_xy,
                                 r, z, r_apex, base_w, k, b, origin, unit, perp,
                                 invert, avg, asym)

    return {
        'ok': True,
        'method': chosen,
        'method_label': ANGLE_METHODS[chosen],
        'requested_method': method,
        'window_frac': window_frac,
        'v_frac': v_frac,
        'left': lf,
        'right': rf,
        'avg_angle': avg,
        'asymmetry': asym,
        'confidence': conf,
        'baseline': {'k': k, 'b': b, 'invert': invert},
        'base_width': base_w,
        'height': height,
        'r_apex': r_apex,
        'comparison': comparison,
        'profile_rz': np.stack([r, z], axis=1),
        'contacts': {'left': (float(left_xy[0]), float(left_xy[1])),
                     'right': (float(right_xy[0]), float(right_xy[1]))},
        'frame': {'origin': origin, 'unit': unit, 'perp': perp},
        'vis': vis,
    }


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def annotate_contact_angle(bgr, method, lf, rf, left_xy, right_xy, r, z,
                           r_apex, base_w, k, b, origin, unit, perp, invert,
                           avg, asym) -> np.ndarray:
    """结果叠加图：轮廓点 + 左右拟合曲线 + 切线 + 接触点 + 角度。"""
    vis = bgr.copy()
    h, w = vis.shape[:2]

    # 轮廓点
    scale = max(1.0, w / 1200.0)
    Pxy = _rz_to_xy(np.stack([r, z], axis=1), origin, unit, perp)
    for (x, y) in Pxy:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            vis[yi, xi] = (60, 220, 60)

    # 拟合曲线 + 切线
    colors = {'left': (60, 80, 255), 'right': (255, 170, 40)}
    for side, f, cxy in (('left', lf, left_xy), ('right', rf, right_xy)):
        col = colors[side]
        if f['ok'] and f['model'] is not None:
            curve = _sample_curve(f['model'], side, f.get('z_hi', z.max()))
            if curve is not None and len(curve) >= 2:
                cxy_img = _rz_to_xy(curve, origin, unit, perp)
                pts = cxy_img.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, col, max(2, int(2 * scale)), cv2.LINE_AA)
            # 切线
            th = math.radians(f['angle_deg'])
            L = 0.28 * base_w
            dr = math.cos(th) * (1 if side == 'left' else -1)
            r0 = f['r0'] if f['r0'] is not None else (
                _xy_to_rz(cxy[None, :], origin, unit, perp)[0, 0])
            t_r, t_z = r0 + dr * L, math.sin(th) * L
            seg = _rz_to_xy(np.array([[r0, 0.0], [t_r, t_z]]), origin, unit, perp)
            cv2.line(vis, tuple(np.round(seg[0]).astype(int)),
                     tuple(np.round(seg[1]).astype(int)), col, max(2, int(2 * scale)),
                     cv2.LINE_AA)
        cx, cy = int(round(cxy[0])), int(round(cxy[1]))
        cv2.circle(vis, (cx, cy), int(7 * scale), col, max(2, int(2 * scale)), cv2.LINE_AA)
        cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS,
                       int(16 * scale), max(1, int(2 * scale)))
        tag = f"{'L' if side == 'left' else 'R'}: " + (
            f"{f['angle_deg']:.1f}°" if f['angle_deg'] is not None else 'fail')
        cv2.putText(vis, tag, (cx + int(12 * scale), cy - int(10 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, col,
                    max(1, int(2 * scale)), cv2.LINE_AA)

    # 基线
    cv2.line(vis, (0, int(round(b))), (w - 1, int(round(k * (w - 1) + b))),
             (80, 80, 255), max(2, int(2 * scale)), cv2.LINE_AA)

    txt = [
        f'contact angle [{ANGLE_METHODS.get(method, method)}]',
        f'left {lf["angle_deg"]:.2f}°   right {rf["angle_deg"]:.2f}°' if (
            lf['angle_deg'] is not None and rf['angle_deg'] is not None)
        else 'one side failed',
        f'average {avg:.2f}°' + (f'   asymmetry {asym:.2f}°' if asym is not None else ''),
    ]
    yy = int(34 * scale)
    for t in txt:
        cv2.putText(vis, t, (int(14 * scale), yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68 * scale, (245, 245, 245), max(1, int(1.6 * scale)), cv2.LINE_AA)
        yy += int(27 * scale)
    return vis


def _comparison_chart(comparison: List[Dict]) -> Optional[np.ndarray]:
    """把各方法的平均接触角画成条形图（供 UI 展示）。"""
    rows = [c for c in comparison if c.get('ok') and c.get('avg') is not None]
    if not rows:
        return None
    W, H = 720, 60 + 40 * len(rows)
    canvas = np.full((H, W, 3), (24, 27, 32), np.uint8)
    cv2.putText(canvas, 'method comparison - average contact angle (deg)',
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 230, 240), 2, cv2.LINE_AA)
    max_a = max(c['avg'] for c in rows)
    max_a = max(max_a, 1.0)
    x0, xmax = 260, W - 80
    for i, c in enumerate(rows):
        y = 58 + 40 * i
        cv2.putText(canvas, c['label'][:26], (12, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 210, 220), 1, cv2.LINE_AA)
        x1 = int(x0 + (xmax - x0) * min(c['avg'] / 180.0, 1.0))
        cv2.rectangle(canvas, (x0, y), (x1, y + 20), (80, 170, 255), -1)
        cv2.putText(canvas, f"{c['avg']:.1f}", (x1 + 6, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 245, 250), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# 供 UI 使用的完整流水线：预处理 → 接触线（可选）→ 接触角拟合
# --------------------------------------------------------------------------- #
def process_contact_angle(bgr: np.ndarray, *,
                          use_sr: bool = False,
                          sr_scale: float = 3.0,
                          method: str = 'young_laplace',
                          window_frac: float = 0.25,
                          v_frac: float = 1.0,
                          compare: bool = True,
                          loc: Optional[Dict] = None,
                          contact: Optional[Dict] = None,
                          contact_fn=None) -> Dict:
    """预处理 + 接触角拟合流水线，输出可直接交给 UI 展示的各阶段图像。

    若外部已传入 `loc` / `contact`（例如来自前面几步），则直接复用，
    避免重复计算；否则内部自行定位并求接触点。

    `contact_fn(work, loc)` 是可选回调：在定位完成后调用，返回接触线结果
    （例如第 2 步 Sobel-lemma 的 `detect_contact_line` 输出），用于替换
    内部的接触点估计，从而让整条流水线使用统一的接触点。
    """
    try:
        import preprocess as pp  # type: ignore
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

    if loc is None:
        loc = pp.locate_drop(work)
        stages.append(('液滴定位（预处理）', pp.annotate(work, loc),
                       str(loc['bbox']) if loc.get('ok') else '定位失败'))
    else:
        stages.append(('液滴定位（预处理参考）', pp.annotate(work, loc),
                       str(loc.get('bbox', '')) if loc.get('ok') else '定位失败'))

    if contact is None and contact_fn is not None:
        try:
            contact = contact_fn(work, loc)
        except Exception:
            contact = None
    if contact is not None and contact.get('ok') and contact.get('vis') is not None:
        stages.append(('接触点（Sobel-lemma）', contact['vis'],
                       f"L({contact['left']['x']:.1f},{contact['left']['y']:.1f}) "
                       f"R({contact['right']['x']:.1f},{contact['right']['y']:.1f})"))

    result = detect_contact_angle(work, loc, contact, method=method,
                                  window_frac=window_frac, v_frac=v_frac,
                                  compare=compare)

    if result.get('ok'):
        # 轮廓 + 拟合曲线图（去文字版，便于看清模型）
        stages.append((f'拟合曲线 · {result["method_label"]}', result['vis'],
                       f'平均 {result["avg_angle"]:.2f}°'))
        if result.get('comparison'):
            chart = _comparison_chart(result['comparison'])
            if chart is not None:
                stages.append(('方法对比', chart, '各方法平均接触角（度）'))
    else:
        stages.append(('接触角拟合失败', work.copy(), result.get('error', '')))

    return {'stages': stages, 'result': result, 'sr_active': sr_ok,
            'ok': result.get('ok', False)}
