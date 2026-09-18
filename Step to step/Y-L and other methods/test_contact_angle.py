"""Y-L / 几何拟合的合成测试（TDD 风格）。

用解析球冠（θ 已知）与 Young–Laplace 数值生成的轮廓作为真值，
检查各拟合方法能否恢复接触角。
运行：python test_contact_angle.py
"""
import math

import cv2
import numpy as np

import contact_angle as ca


def _synth_cap(theta_deg: float, W=800, H=620, R=200.0):
    """生成接触角为 theta 的球冠图像 + 掩码 + loc。基线 y=500，倾斜 0。"""
    th = math.radians(theta_deg)
    cx = W / 2.0
    cy = 500.0 + R * math.cos(th)          # 圆心：接触点处半径与基线夹角 = theta
    # 圆与基线 y=500 的交点
    off = math.sqrt(max(R * R - (500.0 - cy) ** 2, 0.0))
    # 只画液滴一侧（上方）
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(R), 255, -1)
    mask[501:, :] = 0
    img = np.full((H, W, 3), 30, np.uint8)
    img[mask > 0] = (185, 185, 185)
    rng = np.random.default_rng(1)
    img = np.clip(img.astype(int) + rng.integers(-5, 6, (H, W, 1)), 0, 255).astype(np.uint8)
    loc = {'ok': True, 'mask': mask, 'k': 0.0, 'b': 500.0, 'invert': False,
           'bbox': [int(cx - off), int(cy - R), int(cx + off), 500],
           'area': int((mask > 0).sum()), 'score': 1.0}
    return img, loc


def _synth_yl(b: float, lam: float, W=900, H=700):
    """用 Young–Laplace ODE 生成轮廓，反投影成图像与掩码。

    返回 (img, loc, true_angle_deg)；真值由生成参数在 D=1 处读取。
    """
    prof = ca.young_laplace_profile(b, lam, d_max=1.0, n=800)
    assert prof is not None, 'YL 生成失败'
    X, D, phi = prof
    true_ang = math.degrees(float(phi[int(np.argmin(np.abs(D - 1.0)))]))
    Hpx = 300.0
    r_apex = W / 2.0
    z_apex = 120.0                     # 顶点像 y
    # 顶点取 D=0 处；接触在 D=1
    xs_r = r_apex + X * Hpx
    ys = z_apex + D * Hpx
    xs_l = r_apex - X * Hpx
    # 多边形：右轮廓自上而下 + 左轮廓自下而上（镜像）
    poly_r = np.stack([xs_r, ys], axis=1)
    poly_l = np.stack([xs_l[::-1], ys[::-1]], axis=1)
    poly = np.concatenate([poly_r, poly_l]).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    # 基线在 ys 最大处（顶点下方 Hpx）
    base_y = z_apex + Hpx
    mask[int(round(base_y)) + 1:, :] = 0
    img = np.full((H, W, 3), 30, np.uint8)
    img[mask > 0] = (185, 185, 185)
    rng = np.random.default_rng(2)
    img = np.clip(img.astype(int) + rng.integers(-4, 5, (H, W, 1)), 0, 255).astype(np.uint8)
    loc = {'ok': True, 'mask': mask, 'k': 0.0, 'b': float(base_y), 'invert': False,
           'bbox': [int(xs_l.min()), int(ys.min()), int(xs_r.max()), int(base_y)],
           'area': int((mask > 0).sum()), 'score': 1.0}
    return img, loc, true_ang


def test_cap_methods():
    print('== 球冠 theta=90 ==')
    img, loc = _synth_cap(90.0)
    for m in ['circle', 'ellipse', 'poly2', 'poly3', 'spline', 'line',
              'young_laplace_cap', 'young_laplace', 'auto']:
        res = ca.detect_contact_angle(img, loc, None, method=m)
        if res['ok']:
            print(f"  {m:20s} avg={res['avg_angle']:7.2f}  "
                  f"L={res['left']['angle_deg']:7.2f} R={res['right']['angle_deg']:7.2f}")
        else:
            print(f"  {m:20s} FAIL: {res.get('error')}")


def test_cap_angles():
    for theta in (40.0, 90.0, 130.0):
        img, loc = _synth_cap(theta)
        res = ca.detect_contact_angle(img, loc, None, method='circle')
        got = res['avg_angle'] if res['ok'] else float('nan')
        print(f'== 球冠 theta={theta:5.1f} -> circle {got:7.2f} (误差 {got-theta:+.2f}) ==')


def test_yl_recovery():
    # 用带重力的 YL 生成轮廓，再用 YL 反演接触角
    for b, lam in [(1.5, 0.3), (1.8, 0.15), (1.2, 0.0)]:
        img, loc, true_ang = _synth_yl(b, lam)
        res = ca.detect_contact_angle(img, loc, None, method='young_laplace')
        if res['ok']:
            print(f'== YL b={b} lam={lam}: true={true_ang:.2f}  '
                  f'fit avg={res["avg_angle"]:.2f} (误差 {res["avg_angle"]-true_ang:+.2f})  '
                  f'b_fit={res["left"]["model"]["b"]:.3f} lam_fit={res["left"]["model"]["lam"]:.3f}')
        else:
            print('YL FAIL', res.get('error'))


if __name__ == '__main__':
    test_cap_methods()
    test_cap_angles()
    test_yl_recovery()
