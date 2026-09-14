"""针对 Sobel-lemma 接触点定位改进的测试（TDD）。

测试 1：`_contour_contacts` 在合成球冠液滴上，找到的左右接触点应接近
        圆与基线的几何交点。
测试 2：`validate_contacts` 对不合理的接触点（span 过小/过大、不对称）
        正确标记为无效。
"""
import math

import cv2
import numpy as np

import sobel_lemma


def make_spherical_cap(w=300, h=200, cx=150.0, cy=120.0, R=80.0, b=160.0):
    """合成球冠液滴掩码：圆心(cx,cy)、半径 R，液滴在基线 y=b 上方。

    基线水平（k=0），液滴一侧是 y < b（图像上方）。接触点 = 圆与 y=b 的交点：
        x = cx ± sqrt(R^2 - (b - cy)^2)
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
    above = yy < b
    mask[(inside & above)] = 255
    return mask


def _ground_truth_x(cx, cy, R, b):
    dx = math.sqrt(max(0.0, R ** 2 - (b - cy) ** 2))
    return cx - dx, cx + dx


def test_contour_contacts_on_spherical_cap():
    cx, cy, R, b = 150.0, 120.0, 80.0, 160.0
    mask = make_spherical_cap(cx=cx, cy=cy, R=R, b=b)
    lx_true, rx_true = _ground_truth_x(cx, cy, R, b)

    left, right = sobel_lemma._contour_contacts(
        mask, k=0.0, b=b, invert=False, clearance=3.0, contact_band=10.0)

    assert left is not None and right is not None, '应该找到左右两个接触点'
    # 左接触点 x 接近 lx_true，右接近 rx_true（允许亚像素/像素级误差）
    assert abs(left[0] - lx_true) < 3.0, f'左接触点 x={left[0]:.2f}，期望≈{lx_true:.2f}'
    assert abs(right[0] - rx_true) < 3.0, f'右接触点 x={right[0]:.2f}，期望≈{rx_true:.2f}'
    # 接触点应落在基线上（y 接近 b）
    assert abs(left[1] - b) < 2.0 and abs(right[1] - b) < 2.0


def test_validate_contacts_rejects_degenerate_span():
    # span 为 0（左右重合）应该被标记无效
    left = (100.0, 160.0)
    right = (100.0, 160.0)
    ok, warnings = sobel_lemma.validate_contacts(left, right, image_w=300, mask_w=140)
    assert ok is False
    assert any('span' in w.lower() for w in warnings)


def test_validate_contacts_accepts_small_separate_span():
    # 底部尖液滴：左右接触点分开（span 小但 > 1），不应被判为"过小"
    left = (100.0, 160.0)
    right = (116.0, 160.0)  # span=16，左右分开
    ok, warnings = sobel_lemma.validate_contacts(left, right, image_w=1706, mask_w=366)
    assert ok is True, f'左右分开的小 span 不该判失败: {warnings}'


def test_contour_contacts_handles_float_noise():
    # 模拟真实图：掩码 = z > clearance 平切 + x 范围限制，底部是 z 等值线
    # （z 值因 float32 计算有 ~0.001 的噪声，接触点应是底边的左右端点）。
    h, w = 1200, 1700
    k, b = 0.004, 500.0
    zm = sobel_lemma._zmap(k, b, w, h, False)
    xs = np.arange(w)
    inside_x = (xs >= 600) & (xs <= 1100)
    mask = ((zm > 3.0) & inside_x[None, :]).astype(np.uint8) * 255
    left, right = sobel_lemma._contour_contacts(
        mask, k, b, invert=False, clearance=3.0, contact_band=10.0)
    assert left is not None and right is not None, '应该找到左右接触点'
    assert abs(left[0] - 600) < 8, f'左接触点 x={left[0]:.1f} 期望≈600'
    assert abs(right[0] - 1100) < 8, f'右接触点 x={right[0]:.1f} 期望≈1100'


def test_detect_contact_line_uses_contour_contacts():
    # 集成：detect_contact_line 应该用掩码轮廓交点，接触点接近几何真值
    cx, cy, R, b = 150.0, 120.0, 80.0, 160.0
    mask = make_spherical_cap(cx=cx, cy=cy, R=R, b=b)
    h, w = mask.shape
    bgr = np.zeros((h, w, 3), np.uint8)
    bgr[mask > 0] = 128
    loc = {'ok': True, 'mask': mask, 'k': 0.0, 'b': b, 'invert': False}

    r = sobel_lemma.detect_contact_line(bgr, loc, mode='sobel')
    assert r.get('ok'), f'检测失败: {r.get("error")}'
    lx_true, rx_true = _ground_truth_x(cx, cy, R, b)
    assert abs(r['left']['x'] - lx_true) < 3.0, \
        f'左接触点 x={r["left"]["x"]:.2f} 期望≈{lx_true:.2f}'
    assert abs(r['right']['x'] - rx_true) < 3.0, \
        f'右接触点 x={r["right"]["x"]:.2f} 期望≈{rx_true:.2f}'


if __name__ == '__main__':
    import sys
    fns = [f for name, f in sorted(globals().items())
           if name.startswith('test_') and callable(f)]
    for f in fns:
        f()
        print(f'PASS  {f.__name__}')
    print(f'\n{fns.__len__()} tests passed')
