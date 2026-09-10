#!/usr/bin/env python3
"""
OpenDrop 接触角测量 — 结果可视化
================================
为每张图片绘制：
  - 红色基线（表面线）
  - 绿色液滴边缘点
  - 蓝色左右接触点
  - 黄色拟合弧线
  - 角度标注文字

输出: 结果图片/001.jpg ~ 176.jpg
"""

import os, sys, time, math, warnings, importlib.util
import numpy as np
import cv2

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(_SCRIPT_DIR, '图片', '图片')
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, '结果图片')

# ---- 加载 batch_opendrop 模块 ----
spec = importlib.util.spec_from_file_location(
    "batch_opendrop",
    os.path.join(_SCRIPT_DIR, "batch_opendrop.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 引用所有需要的函数
enhance_image = mod.enhance_image
detect_baseline = mod.detect_baseline
extract_contact_angle_features = mod.extract_contact_angle_features
contact_angle_fit_opendrop = mod.contact_angle_fit_opendrop
line_fit = mod.line_fit
circle_fit = mod.circle_fit
rotation_mat2d = mod.rotation_mat2d
Line2 = mod.Line2
Vector2 = mod.Vector2
PI = mod.PI


# ===========================================================================
# 可视化绘制
# ===========================================================================
def draw_results(image, baseline, drop_points, fit_result, filename, idx):
    """
    在图像上叠加可视化结果。

    返回带标注的 BGR 图像。
    """
    h, w = image.shape[:2]
    vis = image.copy()

    # --- 1. 绘制基线 (红色实线) ---
    pt0 = (int(baseline.pt0.x), int(baseline.pt0.y))
    pt1 = (int(baseline.pt1.x), int(baseline.pt1.y))
    cv2.line(vis, pt0, pt1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # --- 2. 绘制液滴边缘点 (绿色小点) ---
    if drop_points.shape[1] > 0:
        pts = drop_points.astype(np.int32).T
        for px, py in pts:
            cv2.circle(vis, (int(px), int(py)), radius=1, color=(0, 255, 0), thickness=-1)

    # --- 3. 绘制左右接触点 (蓝色大圆 + 十字) ---
    left_contact = fit_result.get('left_contact')
    right_contact = fit_result.get('right_contact')

    for contact, label in [(left_contact, 'L'), (right_contact, 'R')]:
        if contact is None:
            continue
        cx, cy = int(contact.x), int(contact.y)
        # 蓝色实心圆
        cv2.circle(vis, (cx, cy), radius=8, color=(255, 0, 0), thickness=-1)
        cv2.circle(vis, (cx, cy), radius=10, color=(255, 255, 255), thickness=2)
        # 十字线
        cv2.line(vis, (cx - 12, cy), (cx + 12, cy), color=(255, 0, 0), thickness=2)
        cv2.line(vis, (cx, cy - 12), (cx, cy + 12), color=(255, 0, 0), thickness=2)

    # --- 4. 绘制拟合弧线 (黄色) ---
    # 从 drop_points 中分离左右侧点（在基线坐标下）
    baseline_np = baseline
    Q = np.array([baseline_np.unit, baseline_np.perp])
    xy = drop_points.astype(float)
    rz = Q @ (xy - np.reshape(baseline_np.pt0, (2, 1)))
    if abs(rz[1].min()) > abs(rz[1].max()):
        Q[1] *= -1
        rz[1] *= -1

    rc = rz[0].mean()
    left_mask = rz[0] < rc
    right_mask = ~left_mask

    # 绘制左侧拟合弧线
    _draw_arc(vis, baseline_np, fit_result.get('left_angle'),
              fit_result.get('left_curvature'),
              fit_result.get('left_contact'),
              fit_result.get('left_arc_center'),
              color=(0, 255, 255))

    # 绘制右侧拟合弧线
    _draw_arc(vis, baseline_np, fit_result.get('right_angle'),
              fit_result.get('right_curvature'),
              fit_result.get('right_contact'),
              fit_result.get('right_arc_center'),
              color=(0, 255, 255))

    # --- 5. 文字标注 ---
    left_deg = math.degrees(fit_result['left_angle']) if fit_result['left_angle'] else None
    right_deg = math.degrees(fit_result['right_angle']) if fit_result['right_angle'] else None

    # 左上角: 序号 + 文件名
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, min(1.2, w / 1200.0))
    thick = max(1, int(font_scale * 2))

    label = f"#{idx:03d}  {filename}"
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thick)
    # 半透明背景
    overlay = vis.copy()
    cv2.rectangle(overlay, (10, 10), (14 + tw, 14 + th + 8), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    cv2.putText(vis, label, (12, 12 + th), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

    # 角度标注
    y_offset = 12 + th + 12
    line_h = th + 8

    if left_deg is not None:
        text = f"Left:  {left_deg:.2f}"
        overlay = vis.copy()
        (tw2, _), _ = cv2.getTextSize(text, font, font_scale * 0.85, thick)
        cv2.rectangle(overlay, (10, y_offset), (14 + tw2, y_offset + line_h + 4), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        cv2.putText(vis, text, (12, y_offset + th), font, font_scale * 0.85, (100, 200, 255), thick, cv2.LINE_AA)
        y_offset += line_h + 6

    if right_deg is not None:
        text = f"Right: {right_deg:.2f}"
        overlay = vis.copy()
        (tw2, _), _ = cv2.getTextSize(text, font, font_scale * 0.85, thick)
        cv2.rectangle(overlay, (10, y_offset), (14 + tw2, y_offset + line_h + 4), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        cv2.putText(vis, text, (12, y_offset + th), font, font_scale * 0.85, (100, 255, 200), thick, cv2.LINE_AA)
        y_offset += line_h + 6

    if left_deg is not None and right_deg is not None:
        avg = (left_deg + right_deg) / 2
        text = f"Avg:   {avg:.2f}"
        overlay = vis.copy()
        (tw2, _), _ = cv2.getTextSize(text, font, font_scale * 0.85, thick)
        cv2.rectangle(overlay, (10, y_offset), (14 + tw2, y_offset + line_h + 4), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        cv2.putText(vis, text, (12, y_offset + th), font, font_scale * 0.85, (255, 255, 100), thick, cv2.LINE_AA)

    return vis


def _draw_arc(vis, baseline, angle, curvature, contact, arc_center, color=(0, 255, 255)):
    """绘制拟合弧线（圆或直线在接触点附近的一段）"""
    if angle is None or contact is None:
        return

    Q = np.array([baseline.unit, baseline.perp])
    pt0_arr = np.array([baseline.pt0.x, baseline.pt0.y])
    contact_arr = np.array([contact.x, contact.y])
    contact_rz = Q @ (contact_arr - pt0_arr)
    cx_rz = contact_rz[0]
    contact_img = contact_arr

    thick = 2

    if curvature is not None and abs(curvature) > 1e-9 and arc_center is not None:
        # 圆形弧线
        radius = 1.0 / abs(curvature)
        center = np.array([arc_center.x, arc_center.y])
        center_rz = Q @ (center - pt0_arr)

        # 从接触点沿弧线方向画一小段
        if center_rz[1] < radius:
            # 弧线与基线相交的角度范围
            half_angle = math.acos(center_rz[1] / radius) if radius > 0 else 0
            q0 = math.atan2(center_rz[1], center_rz[0] - contact_rz[0])
            arc_span = min(half_angle * 0.6, math.radians(30))  # 最多画30度

            if curvature < 0:
                start_angle = q0 - arc_span
                end_angle = q0
            else:
                start_angle = q0
                end_angle = q0 + arc_span

            # 生成弧线点
            n_pts = 30
            angles = np.linspace(start_angle, end_angle, n_pts)
            arc_x = center[0] + radius * np.cos(angles)
            arc_y = center[1] + radius * np.sin(angles)

            pts = np.column_stack([arc_x, arc_y]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=thick, lineType=cv2.LINE_AA)
    else:
        # 直线段
        angle_rad = angle
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])
        length = 40
        pt1 = contact_img - direction * length * 0.5
        pt2 = contact_img + direction * length * 1.5
        cv2.line(vis, tuple(pt1.astype(int)), tuple(pt2.astype(int)),
                 color=color, thickness=thick, lineType=cv2.LINE_AA)


# ===========================================================================
# 单张处理 + 可视化
# ===========================================================================
def process_and_visualize(image_path, idx, total):
    """处理单张图片并返回可视化图像"""
    basename = os.path.basename(image_path)
    print(f"[{idx}/{total}] {basename} ...", end=" ", flush=True)

    try:
        # 读取
        with open(image_path, 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            print("[FAIL] read")
            return None

        h, w = img.shape[:2]
        enhanced = enhance_image(img)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # 基线检测
        k, b_val = detect_baseline(gray, h, w)
        baseline = Line2(Vector2(0.0, k * 0 + b_val),
                         Vector2(float(w - 1), k * (w - 1) + b_val))

        # 特征提取
        drop_points = extract_contact_angle_features(
            enhanced, baseline, inverted=False, thresh=0.3,
        )

        if drop_points.shape[1] < 10:
            # 即使没有检测到足够点，也画出基线和基本信息
            vis = img.copy()
            pt0 = (int(baseline.pt0.x), int(baseline.pt0.y))
            pt1 = (int(baseline.pt1.x), int(baseline.pt1.y))
            cv2.line(vis, pt0, pt1, color=(0, 0, 255), thickness=2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = max(0.7, min(1.2, w / 1200.0))
            cv2.putText(vis, f"#{idx:03d} {basename} [NO EDGE]", (12, 40),
                        font, fs, (0, 0, 255), 2, cv2.LINE_AA)
            print("[WARN] no edge")
            return vis

        # 接触角拟合
        fit = contact_angle_fit_opendrop(drop_points, baseline)

        # 绘制可视化
        vis = draw_results(enhanced, baseline, drop_points, fit, basename, idx)

        left_deg = math.degrees(fit['left_angle']) if fit['left_angle'] else None
        right_deg = math.degrees(fit['right_angle']) if fit['right_angle'] else None
        parts = []
        if left_deg is not None: parts.append(f"L={left_deg:.1f}")
        if right_deg is not None: parts.append(f"R={right_deg:.1f}")
        print(f"[OK] {' / '.join(parts)}")

        return vis

    except Exception as e:
        print(f"[FAIL] {e}")
        return None


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted([f for f in os.listdir(IMAGE_DIR)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
    if not files:
        print("[ERROR] No images"); return

    total = len(files)
    print(f"{'=' * 60}")
    print(f"  OpenDrop 结果可视化")
    print(f"  图片数量: {total}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    saved = 0

    for idx, fn in enumerate(files, 1):
        img_path = os.path.join(IMAGE_DIR, fn)
        vis = process_and_visualize(img_path, idx, total)

        if vis is not None:
            # 严格按照序号命名: 001.jpg, 002.jpg, ...
            out_name = f"{idx:03d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            # 使用 imencode 避免中文路径问题
            success, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                with open(out_path, 'wb') as f:
                    f.write(buf.tobytes())
                saved += 1

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  完成! 耗时 {elapsed:.1f}s | 保存 {saved}/{total} 张图片")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
