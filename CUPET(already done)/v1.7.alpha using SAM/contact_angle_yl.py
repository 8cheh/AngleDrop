"""
接触角测量 - 完整流水线 (新文件)
================================
1. 图像增强          -> CLAHE + 双边滤波降噪 + 锐化
2. AI 分割 (SAM)     -> 仅分割液滴 (基底用传统 CV 检测)
3. 基准线拟合        -> Sobel梯度边缘检测 + RANSAC (基底上边界)
4. 接触点定位        -> 液滴轮廓与基准线的几何交点
5. 角度计算与可视化  -> Young-Laplace 方程数值拟合

策略：SAM 专攻液滴分割，基底用传统图像处理检测明暗交界
用法：python contact_angle_yl.py [图片路径]
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import os
import sys


# ============================================================
# 命令行参数
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="接触角测量 - Young-Laplace 方法")
    p.add_argument("image", nargs="?", default="drop_image.jpg", help="输入图片路径")
    p.add_argument("--sam", default="sam_vit_h_4b8939.pth", help="SAM 权重路径")
    p.add_argument("--output", "-o", default=None, help="保存结果图路径")
    p.add_argument("--no-display", action="store_true", help="不显示 GUI")
    p.add_argument("--debug", action="store_true", help="保存中间调试图片")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X","Y","W","H"), help="ROI裁剪")
    return p.parse_args()


# ============================================================
# 主类
# ============================================================
class ContactAngleAnalyzer:
    """接触角分析器

    流水线:
      步骤1 - 图像增强 (CLAHE + 降噪 + 锐化)
      步骤2 - SAM 液滴分割 (AI)
      步骤3 - 基准线检测 (传统CV: Sobel + RANSAC)
      步骤4 - 接触点定位 (轮廓-基线几何交点)
      步骤5 - Young-Laplace 拟合 + 角度计算
    """

    def __init__(self, sam_ckpt: str, device: str = None):
        print("=" * 60)
        print("  接触角分析器 · Contact Angle Analyzer")
        print("  Young-Laplace Method")
        print("=" * 60)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[初始化] 设备: {self.device}")

        if not os.path.exists(sam_ckpt):
            raise FileNotFoundError(f"SAM 权重不存在: {sam_ckpt}")

        print("[初始化] 加载 SAM vit_h ...")
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
        sam.to(device=self.device)
        sam.eval()

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        print("[初始化] SAM 就绪\n")

    # ============================================================
    # 主入口
    # ============================================================
    def analyze(self, image_path: str, roi=None, debug: bool = False) -> dict:
        print(f"[流水线] {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取: {image_path}")
        if roi:
            x, y, w, h = roi
            img = img[y:y+h, x:x+w]
            print(f"[流水线] ROI: ({x},{y},{w},{h})")
        h, w = img.shape[:2]
        print(f"[流水线] 尺寸: {w} x {h}")

        # ======== 步骤1: 图像增强 ========
        img_enhanced = self._step1_enhance(img)
        if debug:
            cv2.imwrite("debug_01_enhanced.jpg", img_enhanced)

        # ======== 步骤2: SAM 液滴分割 ========
        droplet_mask = self._step2_segment_droplet(img_enhanced, debug)
        if debug:
            cv2.imwrite("debug_02_droplet_mask.png", droplet_mask)

        # ======== 步骤3: 基准线检测 (传统CV) ========
        baseline = self._step3_detect_baseline(img, droplet_mask)
        if baseline is None:
            raise RuntimeError("基准线检测失败!")

        # ======== 步骤4: 接触点定位 ========
        contour, left_pt, right_pt, apex = self._step4_contact_points(
            droplet_mask, baseline)
        if left_pt is None or right_pt is None:
            raise RuntimeError("接触点定位失败!")

        # ======== 步骤5: Young-Laplace 拟合 ========
        yl_result = self._step5_young_laplace(
            contour, baseline, left_pt, right_pt, apex)

        # ======== 可视化 ========
        self._visualize(img, img_enhanced, droplet_mask,
                        baseline, contour, left_pt, right_pt,
                        apex, yl_result)

        return {
            "left_angle": yl_result["left_angle"],
            "right_angle": yl_result["right_angle"],
            "average_angle": (yl_result["left_angle"] + yl_result["right_angle"]) / 2,
            "yl_params": yl_result.get("params", None),
            "apex": apex,
            "baseline": baseline,
            "contact_points": (left_pt, right_pt),
        }

    # ============================================================
    # 步骤 1: 图像增强
    # ============================================================
    def _step1_enhance(self, img: np.ndarray) -> np.ndarray:
        """多阶段增强: 降噪 → CLAHE → 锐化"""
        print("[步骤1] 图像增强...")

        # 1a. 双边滤波降噪（保边）
        denoised = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)

        # 1b. CLAHE 对比度增强（LAB 色彩空间，仅 L 通道）
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        # 1c. Unsharp Masking 锐化
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.5)
        enhanced = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        print("[步骤1] 完成 ✓")
        return enhanced

    # ============================================================
    # 步骤 2: SAM 液滴分割
    # ============================================================
    def _step2_segment_droplet(self, img: np.ndarray,
                                debug: bool = False) -> np.ndarray:
        """SAM 自动分割 → 多维评分筛选液滴掩码

        SAM 只负责分割液滴；
        基底（均匀亮区）在步骤3用传统CV检测。
        """
        print("[步骤2] SAM 液滴分割...")
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = self.mask_generator.generate(img_rgb)
        print(f"  SAM 产生 {len(masks)} 个候选掩码")

        # ---- 对每个掩码评分 ----
        scored = []
        for idx, m in enumerate(masks):
            seg = m['segmentation']
            area = m['area']
            bx, by, bw, bh = m['bbox']
            cx = bx + bw / 2
            cy = by + bh / 2

            # ---- 硬性排除 ----
            if by < 3 and bh < 50:      # 顶部噪声
                continue
            if by + bh > h * 0.85 and bw > w * 0.6:  # 基底
                continue
            if area < 200:               # 太小
                continue
            if area > w * h * 0.7:       # 几乎占满全图
                continue

            # 轮廓 + 形状指标
            cnts, _ = cv2.findContours(
                seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue

            circularity = 4 * np.pi * area / (peri * peri)
            aspect = bw / max(bh, 1)
            solidity = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1)

            # ---- 综合评分 ----
            score = 0.0

            # 位置分：液滴应在图像中上部 (y 比率 0.20~0.65)
            y_frac = cy / h
            if 0.20 < y_frac < 0.65:
                pos_score = 1.0 - abs(y_frac - 0.40) / 0.40
                score += pos_score * 3.0

            # 面积分：中等偏大 (2000~150000 px)
            if 2000 < area < 150000:
                area_score = min(area / 30000, 30000 / max(area, 1))
                area_score = min(area_score, 2.0)
                score += area_score * 2.0

            # 形状分
            if 0.3 < circularity < 0.95:
                score += circularity * 2.0
            if 0.4 < aspect < 3.0:
                score += (1.0 - abs(aspect - 1.2) / 2.0) * 1.5
            if solidity > 0.7:
                score += solidity * 1.0

            # 底部平坦度
            bottom_row = by + bh - 1
            if bottom_row < h - 5:
                strip = seg[max(0, bottom_row-5):bottom_row+1, bx:bx+bw]
                if strip.size > 0:
                    fill = np.mean(strip, axis=1)
                    if len(fill) > 0 and np.max(fill) > 0.3:
                        score += 1.0

            # 边缘惩罚
            edge_pen = 0
            if bx < 3:   edge_pen += 2
            if by < 3:   edge_pen += 2
            if bx + bw > w - 3: edge_pen += 2
            score -= edge_pen

            scored.append((score, idx, m))

        # ---- 回退策略 ----
        if not scored:
            print("  [警告] 无高分候选, 回退...")
            for idx, m in enumerate(masks):
                area = m['area']
                cy2 = m['bbox'][1] + m['bbox'][3] / 2
                if 2000 < area < w * h * 0.6 and 0.15 < cy2 / h < 0.7:
                    scored.append((area, idx, m))
        if not scored:
            for m in sorted(masks, key=lambda x: x['area'], reverse=True):
                if 500 < m['area'] < w * h * 0.5:
                    scored.append((m['area'], 0, m))
                    break

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        best_seg = best[2]['segmentation']

        print(f"  [液滴] #{best[1]} 得分={best[0]:.2f} "
              f"面积={best[2]['area']} bbox={best[2]['bbox']}")

        # ---- 形态学后处理 ----
        droplet = best_seg.astype(np.uint8) * 255

        # 闭运算 → 填充小孔
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        droplet = cv2.morphologyEx(droplet, cv2.MORPH_CLOSE, k1, iterations=2)

        # 仅保留最大连通域
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            droplet, connectivity=8)
        if n_labels > 1:
            largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            droplet = (labels == largest).astype(np.uint8) * 255

        # 填充内部孔洞
        cnts, _ = cv2.findContours(droplet, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        droplet[:] = 0
        if cnts:
            cv2.drawContours(droplet, [max(cnts, key=cv2.contourArea)], -1, 255, -1)

        # 开运算 → 平滑边缘
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        droplet = cv2.morphologyEx(droplet, cv2.MORPH_OPEN, k2)

        print(f"[步骤2] 液滴掩码: {np.sum(droplet > 0)} px ✓")
        return droplet

    # ============================================================
    # 步骤 3: 基准线检测（传统 CV，不依赖 SAM）
    # ============================================================
    def _step3_detect_baseline(self, img: np.ndarray,
                                droplet_mask: np.ndarray) -> dict:
        """传统 CV 检测基底上表面 = 基准线

        原理: 基底是图片下半部明亮均匀区域。
        找到明→暗过渡带 → RANSAC 直线拟合。

        对暗背景 + 中间液滴 + 亮基底这类图像非常可靠。
        """
        print("[步骤3] 基准线检测 (传统CV)...")
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---- 方法: Sobel Y方向梯度 + 列扫描 ----
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_y_abs = np.abs(sobel_y)

        # 限定搜索区域: 从液滴掩码底部到图像底部之间
        # 找到液滴最低点
        if np.sum(droplet_mask) > 0:
            drop_ys = np.where(droplet_mask > 0)[0]
            drop_bottom = np.max(drop_ys)
            search_top = drop_bottom - 10
        else:
            search_top = h // 3
        search_bottom = h - 5

        # 在搜索区域内逐列找最强水平边缘
        edge_pts = []
        for x in range(0, w, 2):
            col = sobel_y_abs[search_top:search_bottom, x]
            if len(col) > 0 and np.max(col) > 40:
                offset = np.argmax(col)
                edge_y = search_top + offset
                edge_pts.append([x, edge_y])

        if len(edge_pts) < 10:
            # 回退: 行均值梯度
            print("  [警告] Sobel 点不足, 使用行均值梯度")
            row_means = np.mean(gray.astype(float), axis=1)
            row_means_s = uniform_filter1d(row_means, size=15)
            row_grad = np.abs(np.gradient(row_means_s))
            best_y = h // 3 + np.argmax(row_grad[h//3:h-20])
            for x in range(0, w, 5):
                edge_pts.append([x, best_y])

        pts = np.array(edge_pts, dtype=np.float32)

        # ---- 离群点过滤 (MAD) ----
        y_vals = pts[:, 1]
        y_med = np.median(y_vals)
        y_mad = np.median(np.abs(y_vals - y_med)) + 1e-8
        keep = np.abs(y_vals - y_med) < max(3 * y_mad, 15)
        pts_in = pts[keep] if np.sum(keep) >= 5 else pts

        # ---- RANSAC 直线拟合 ----
        [vx, vy, x0, y0] = cv2.fitLine(pts_in, cv2.DIST_HUBER, 0, 0.01, 0.99)

        if abs(vx) < 1e-8:
            k = 0.0
        else:
            k = float((vy / vx)[0])
        b_val = float((y0 - k * x0)[0])

        # 限制斜率（基底应接近水平）
        if abs(k) > 0.3:
            print(f"  [警告] 斜率={k:.3f}过大, 强制k=0")
            k = 0.0
            b_val = np.median(pts_in[:, 1])

        result = {
            'k': k,
            'b': b_val,
            'point': (int(x0[0]), int(y0[0])),
            'line_points': ((0, int(b_val)), (w - 1, int(k * w + b_val))),
            'edge_points': pts_in,
        }

        print(f"[步骤3] 基准线: y = {k:.4f}x + {b_val:.1f} "
              f"({len(pts_in)} 边缘点) ✓")
        return result

    # ============================================================
    # 步骤 4: 接触点定位
    # ============================================================
    def _step4_contact_points(self, droplet_mask: np.ndarray,
                               baseline: dict) -> tuple:
        """液滴轮廓与基准线精确求交 → 接触点 + 顶点"""
        print("[步骤4] 接触点定位...")
        h, w = droplet_mask.shape
        k = baseline['k']
        b_val = baseline['b']

        # 提取最大轮廓
        cnts, _ = cv2.findContours(droplet_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None, None, None, None
        cnt = max(cnts, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2).astype(np.float32)

        # ---- 顶点: y 最小 (图像中最高) ----
        apex_idx = np.argmin(pts[:, 1])
        apex = (int(pts[apex_idx][0]), int(pts[apex_idx][1]))

        # ---- 遍历轮廓线段，求与基线的交点 ----
        intersections = []
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            # 快速排除: 两端点同侧
            d1 = p1[1] - (k * p1[0] + b_val)
            d2 = p2[1] - (k * p2[0] + b_val)
            if d1 * d2 > 0:
                continue

            pt = self._seg_line_intersect(p1, p2, k, b_val)
            if pt is None:
                continue
            # 接触点必须在顶点下方
            if pt[1] <= apex[1] + 5:
                continue
            if not (0 <= pt[0] <= w and 0 <= pt[1] <= h):
                continue
            intersections.append(pt)

        # ---- 处理结果 ----
        if len(intersections) < 2:
            print(f"  [警告] 仅 {len(intersections)} 个交点, 使用回退")
            max_y = np.max(pts[:, 1])
            band = max(15, int(h * 0.03))
            bottom = pts[pts[:, 1] >= max_y - band]
            if len(bottom) >= 2:
                left_pt = tuple(bottom[np.argmin(bottom[:, 0])].astype(int))
                right_pt = tuple(bottom[np.argmax(bottom[:, 0])].astype(int))
            elif len(intersections) == 1:
                p0 = intersections[0]
                if len(bottom) > 0:
                    far = bottom[np.argmax(bottom[:, 0])].astype(int)
                else:
                    far = (w - 1, int(k * (w - 1) + b_val))
                if p0[0] < far[0]:
                    left_pt = (int(p0[0]), int(p0[1]))
                    right_pt = (int(far[0]), int(far[1]))
                else:
                    left_pt = (int(far[0]), int(far[1]))
                    right_pt = (int(p0[0]), int(p0[1]))
            else:
                return cnt, None, None, apex
        else:
            # 聚类去重
            intersections.sort(key=lambda p: p[0])
            merged = []
            for pt in intersections:
                if not merged or pt[0] - merged[-1][0] > 20:
                    merged.append(pt)
                else:
                    merged[-1] = ((merged[-1][0]+pt[0])/2, (merged[-1][1]+pt[1])/2)
            if len(merged) >= 2:
                left_pt = (int(merged[0][0]), int(merged[0][1]))
                right_pt = (int(merged[-1][0]), int(merged[-1][1]))
            else:
                return cnt, None, None, apex

        if left_pt[0] > right_pt[0]:
            left_pt, right_pt = right_pt, left_pt

        print(f"[步骤4] 顶点: {apex}, 左: {left_pt}, 右: {right_pt} ✓")
        return cnt, left_pt, right_pt, apex

    @staticmethod
    def _seg_line_intersect(p1, p2, k, b_val):
        """线段 p1→p2 与直线 y=kx+b 的交点"""
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 1e-8:
            xi, yi = x1, k * x1 + b_val
        elif abs(k) > 1e6:
            return None
        else:
            denom = dy - k * dx
            if abs(denom) < 1e-10:
                return None
            t = (k * x1 + b_val - y1) / denom
            if t < -0.01 or t > 1.01:
                return None
            xi = x1 + t * dx
            yi = y1 + t * dy
        return (xi, yi)

    # ============================================================
    # 步骤 5: Young-Laplace 方程拟合
    # ============================================================
    def _step5_young_laplace(self, contour, baseline,
                              left_pt, right_pt, apex) -> dict:
        """Young-Laplace 方程拟合液滴轮廓 → 接触角

        轴对称 Young-Laplace:
          dφ/ds = 2b - c·z - sin(φ)/x
          dx/ds = cos(φ)
          dz/ds = sin(φ)
        b = 1/R₀ (顶点曲率), c = Δρg/γ (毛细常数)
        """
        print("[步骤5] Young-Laplace 拟合...")

        if contour is None or left_pt is None or right_pt is None:
            return {"left_angle": 0.0, "right_angle": 0.0}

        pts = contour.reshape(-1, 2).astype(np.float32)
        k = baseline['k']
        b_val = baseline['b']
        ap = np.array(apex, dtype=np.float32)

        # ---- 物理坐标系: 原点=顶点, z 向上 ----
        pts_p = pts.copy()
        pts_p[:, 0] -= ap[0]
        pts_p[:, 1] = ap[1] - pts_p[:, 1]   # z = apex_y - y

        # 基线在物理坐标中的 z (负值，在顶点下方)
        z_bl = ap[1] - (k * left_pt[0] + b_val)
        z_br = ap[1] - (k * right_pt[0] + b_val)
        z_base = (z_bl + z_br) / 2

        # 左右轮廓分离
        left_c = pts_p[(pts_p[:, 0] < -5) & (pts_p[:, 1] > z_base)]
        right_c = pts_p[(pts_p[:, 0] > 5) & (pts_p[:, 1] > z_base)]

        result = {}
        for side, contour_side in [("left", left_c), ("right", right_c)]:
            if len(contour_side) >= 15:
                ang, params = self._fit_yl_one_side(contour_side, z_base, side)
                result[f"{side}_angle"] = ang
                result[f"params_{side}"] = params
            else:
                result[f"{side}_angle"] = self._poly_fallback(
                    pts_p, left_pt if side == "left" else right_pt,
                    ap, baseline, side)
                result[f"params_{side}"] = None

        print(f"[步骤5] 左={result['left_angle']:.2f}°  "
              f"右={result['right_angle']:.2f}° ✓")
        return result

    def _fit_yl_one_side(self, contour_p, z_base, side) -> tuple:
        """单侧 Young-Laplace 拟合"""
        x_raw = np.abs(contour_p[:, 0])
        z_raw = contour_p[:, 1]

        # 按 z 降序 (顶点→基线)
        order = np.argsort(z_raw)[::-1]
        x_s = x_raw[order]
        z_s = z_raw[order]

        # 去噪
        if len(x_s) > 30:
            wnd = max(5, len(x_s) // 15)
            kern = np.ones(wnd) / wnd
            x_sm = np.convolve(x_s, kern, mode='same')
            dev = np.abs(x_s - x_sm)
            mad = np.median(dev) + 1e-8
            keep = dev < 5 * mad + 8
            x_s, z_s = x_s[keep], z_s[keep]

        if len(x_s) < 10:
            return 0.0, None

        # 初始 b (顶点曲率) 估计
        near = z_s > -8
        if np.sum(near) < 3:
            near = np.ones(len(z_s), dtype=bool)
        xn, zn = x_s[near], z_s[near]
        if len(xn) >= 3:
            A = np.column_stack([zn, np.ones_like(zn)])
            rhs = xn**2 + zn**2
            try:
                sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                zc = sol[0] / 2
                r_est = np.sqrt(max(sol[1] + zc**2, 1.0))
                b_init = 1.0 / max(r_est, 5)
            except np.linalg.LinAlgError:
                b_init = 1.0 / 80.0
        else:
            b_init = 1.0 / 80.0

        b_init = float(np.clip(b_init, 1e-5, 0.5))
        c_init = 5e-4
        z_stop = z_base - 3

        def objective(params):
            b_f = max(float(params[0]), 1e-7)
            c_f = max(float(params[1]), 1e-12)
            try:
                _, x_yl, z_yl, _ = self._solve_yl(b_f, c_f, z_stop)
                if len(z_yl) < 5:
                    return np.full(len(z_s), 500.0)
                x_int = np.interp(z_s, z_yl[::-1], x_yl[::-1],
                                  left=500.0, right=500.0)
                return (x_int - x_s) / max(np.mean(x_s), 1.0)
            except Exception:
                return np.full(len(z_s), 500.0)

        try:
            res = least_squares(objective, [b_init, c_init],
                                bounds=([1e-7, 1e-12], [0.5, 0.5]),
                                method='trf', max_nfev=300,
                                ftol=1e-7, xtol=1e-7,
                                loss='soft_l1', f_scale=0.5)
            b_opt, c_opt = float(res.x[0]), float(res.x[1])
        except Exception:
            b_opt, c_opt = b_init, c_init

        b_opt = max(b_opt, 1e-7)
        c_opt = max(c_opt, 1e-12)

        # 提取接触角
        try:
            _, x_yl, z_yl, phi_yl = self._solve_yl(b_opt, c_opt, z_stop, 500)
            phi_bl = np.interp(z_base, z_yl[::-1], phi_yl[::-1])
            angle = float(np.degrees(abs(phi_bl)))
            angle = np.clip(angle, 0.0, 179.9)
        except Exception:
            angle = 0.0

        return angle, {
            'b': b_opt, 'c': c_opt, 'R0': 1.0 / b_opt,
            'yl_x': x_yl, 'yl_z': z_yl,
        }

    def _solve_yl(self, b, c, z_stop, n_pts=300):
        """数值求解 Young-Laplace ODE"""
        s0, eps = 1e-8, 1e-12
        phi0 = b * s0
        x0_val = s0
        z0_val = (b / 2.0) * s0 * s0

        def ode(s, y):
            x, z_val, phi = y
            sin_div = b if x < eps else np.sin(phi) / x
            dphi = 2.0 * b - c * z_val - sin_div
            return [np.cos(phi), np.sin(phi), dphi]

        def event(s, y):
            return y[1] - z_stop
        event.terminal = True
        event.direction = -1

        sol = solve_ivp(ode, [s0, 8000.0], [x0_val, z0_val, phi0],
                        method='RK45', events=event, dense_output=True,
                        rtol=1e-9, atol=1e-11, max_step=3.0)
        s_ev = np.linspace(s0, sol.t[-1], n_pts)
        y_ev = sol.sol(s_ev)
        return s_ev, y_ev[0], y_ev[1], y_ev[2]

    def _poly_fallback(self, pts_p, contact_pt, ap, baseline, side) -> float:
        """多项式回退: 接触点邻域二次拟合 → 切线角"""
        cx = contact_pt[0] - ap[0]
        k = baseline['k']
        rng = 80

        if side == 'left':
            nb = pts_p[(pts_p[:, 0] < -2) & (np.abs(pts_p[:, 0] - cx) < rng)]
        else:
            nb = pts_p[(pts_p[:, 0] > 2) & (np.abs(pts_p[:, 0] - cx) < rng)]

        if len(nb) < 5:
            return 0.0
        try:
            cfs = np.polyfit(nb[:, 0], nb[:, 1], 2)
            dzdx = np.polyval([2*cfs[0], cfs[1]], cx)
            phi = np.arctan(dzdx)
            bl_ang = np.arctan(k)
            return float(np.clip(abs(np.degrees(phi - bl_ang)), 0, 180))
        except Exception:
            return 0.0

    # ============================================================
    # 可视化: 2×3 布局
    # ============================================================
    def _visualize(self, img_orig, img_enh, droplet, baseline,
                    contour, left_pt, right_pt, apex, yl_res):
        print("[可视化] 生成...")
        h, w = img_orig.shape[:2]
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_enh_rgb = cv2.cvtColor(img_enh, cv2.COLOR_BGR2RGB)
        k, b_val = baseline['k'], baseline['b']

        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.suptitle("Contact Angle Analysis — Young-Laplace Method",
                     fontsize=17, fontweight='bold', y=0.97)

        # ---- (1) 原始 ----
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title("(1) Original Image", fontsize=13, fontweight='bold')
        axes[0, 0].axis('off')

        # ---- (2) 增强 ----
        axes[0, 1].imshow(img_enh_rgb)
        axes[0, 1].set_title("(2) Enhanced (CLAHE+Denoise+Sharpen)",
                             fontsize=13, fontweight='bold')
        axes[0, 1].axis('off')

        # ---- (3) 分割 + 边缘点 ----
        ov = img_rgb.copy().astype(float)
        dm = droplet > 0
        ov[dm] = ov[dm] * 0.35 + np.array([30, 144, 255]) * 0.65
        axes[0, 2].imshow(ov.astype(np.uint8))
        if contour is not None:
            axes[0, 2].plot(contour[:, 0, 0], contour[:, 0, 1], 'lime', lw=2, alpha=0.9)
        if 'edge_points' in baseline:
            ep = baseline['edge_points']
            axes[0, 2].scatter(ep[:, 0], ep[:, 1], c='red', s=3, alpha=0.7)
        axes[0, 2].set_title("(3) SAM Droplet + Substrate Edge",
                             fontsize=13, fontweight='bold')
        axes[0, 2].axis('off')

        # ---- (4) 基准线 + 接触点 + 顶点 ----
        ax = axes[1, 0]
        ax.imshow(img_rgb)
        xl = np.array([0, w - 1])
        yl = k * xl + b_val
        ax.plot(xl, yl, 'r-', lw=3, alpha=0.9, label='Baseline (RANSAC)')
        if contour is not None:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'lime', lw=1.5, alpha=0.8,
                    label='Contour')
        if left_pt:
            ax.plot(left_pt[0], left_pt[1], 'o', color='#00FF00', ms=14,
                    mec='white', mew=2.5, label=f"Left: {yl_res['left_angle']:.1f}°")
        if right_pt:
            ax.plot(right_pt[0], right_pt[1], 'o', color='#00BFFF', ms=14,
                    mec='white', mew=2.5, label=f"Right: {yl_res['right_angle']:.1f}°")
        if apex:
            ax.plot(apex[0], apex[1], '*', color='yellow', ms=16,
                    mec='black', mew=1.5, label='Apex')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
        ax.set_title("(4) Baseline & Contact Points", fontsize=13, fontweight='bold')
        ax.axis('off')

        # ---- (5) YL 拟合 ----
        ax = axes[1, 1]
        ax.imshow(img_rgb)
        if contour is not None:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'white', lw=1, alpha=0.4)
        ap_arr = np.array(apex, dtype=float)
        colors = {'left': '#FF4444', 'right': '#44CCFF'}
        for sn, col in colors.items():
            pk = f'params_{sn}'
            if pk in yl_res and yl_res[pk] is not None:
                p = yl_res[pk]
                if p.get('yl_x') is not None and len(p['yl_x']) > 0:
                    sgn = -1 if sn == 'left' else 1
                    xi = ap_arr[0] + sgn * p['yl_x']
                    yi = ap_arr[1] - p['yl_z']
                    ybl = k * xi + b_val
                    vis = yi < ybl + 10
                    ax.plot(xi[vis], yi[vis], '-', color=col, lw=3, alpha=0.9,
                            label=f'YL fit ({sn})')
        ax.plot(xl, yl, 'r--', lw=2, alpha=0.7)
        if left_pt:
            ax.plot(left_pt[0], left_pt[1], 'o', color='#FF4444', ms=10, mec='white', mew=1.5)
        if right_pt:
            ax.plot(right_pt[0], right_pt[1], 'o', color='#44CCFF', ms=10, mec='white', mew=1.5)
        if apex:
            ax.plot(apex[0], apex[1], 'y*', ms=12, mec='black', mew=1)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
        ax.set_title("(5) Young-Laplace Profile Fit", fontsize=13, fontweight='bold')
        ax.axis('off')

        # ---- (6) 特写 ----
        ax = axes[1, 2]
        if contour is not None and apex is not None:
            bx, by, bw, bh = cv2.boundingRect(contour)
            pad = 40
            x1c = max(0, bx - pad); y1c = max(0, by - pad)
            x2c = min(w, bx + bw + pad); y2c = min(h, by + bh + pad)
            ax.imshow(img_rgb[y1c:y2c, x1c:x2c])

            def tc(p):
                return (p[0] - x1c, p[1] - y1c)

            ax.plot([0, x2c - x1c],
                    [k * x1c + b_val - y1c, k * x2c + b_val - y1c],
                    'r-', lw=2)
            if left_pt:
                lp = tc(left_pt)
                ax.plot(lp[0], lp[1], 'o', color='#00FF00', ms=10, mec='white', mew=2)
                ox = -80 if lp[0] > 80 else 10
                ax.annotate(f"{yl_res['left_angle']:.1f}°", xy=lp,
                            xytext=(lp[0]+ox, lp[1]-35), fontsize=13,
                            color='#00FF00', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='#00FF00',
                                            lw=2, connectionstyle='arc3,rad=-0.3'))
            if right_pt:
                rp = tc(right_pt)
                ax.plot(rp[0], rp[1], 'o', color='#00BFFF', ms=10, mec='white', mew=2)
                ox = 40 if rp[0] < (x2c - x1c - 80) else -80
                ax.annotate(f"{yl_res['right_angle']:.1f}°", xy=rp,
                            xytext=(rp[0]+ox, rp[1]-35), fontsize=13,
                            color='#00BFFF', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='#00BFFF',
                                            lw=2, connectionstyle='arc3,rad=0.3'))
            if apex:
                apc = tc(apex)
                ax.plot(apc[0], apc[1], 'y*', ms=12, mec='black', mew=1)
        ax.set_title("(6) Contact Angle — Detail", fontsize=13, fontweight='bold')
        ax.axis('off')

        # 底部汇总
        avg = (yl_res['left_angle'] + yl_res['right_angle']) / 2
        fig.text(0.5, 0.015,
                 f"Left: {yl_res['left_angle']:.2f}°  |  "
                 f"Right: {yl_res['right_angle']:.2f}°  |  "
                 f"Average: {avg:.2f}°  |  Young-Laplace Method",
                 ha='center', fontsize=15, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.95))

        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        print("[可视化] 完成 ✓")


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.sam):
        print(f"\n❌ SAM 模型不存在: {args.sam}")
        print("   请下载 sam_vit_h_4b8939.pth 放在当前目录")
        print("   https://github.com/facebookresearch/segment-anything")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"\n❌ 图片不存在: {args.image}")
        sys.exit(1)

    try:
        analyzer = ContactAngleAnalyzer(args.sam)
        result = analyzer.analyze(
            args.image,
            roi=tuple(args.crop) if args.crop else None,
            debug=args.debug,
        )

        print("\n" + "=" * 60)
        print("  最终结果")
        print("=" * 60)
        print(f"  左接触角  (Left)  : {result['left_angle']:8.3f}°")
        print(f"  右接触角  (Right) : {result['right_angle']:8.3f}°")
        print(f"  平均      (Avg)   : {result['average_angle']:8.3f}°")
        print("=" * 60)

        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\n[保存] 结果图 → {args.output}")

        if not args.no_display:
            plt.show()

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
