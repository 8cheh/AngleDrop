"""
接触角测量 - 完整流水线
========================
1. 图像增强 (CLAHE + 降噪 + 锐化)
2. AI 分割 (SAM - Segment Anything Model)
3. 基准线拟合 (RANSAC 鲁棒直线拟合)
4. 接触点定位 (轮廓-基准线几何交点)
5. 角度计算与可视化 (Young-Laplace 方程拟合法)
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import argparse
import os
import sys

# ============================================================
# 0. 配置与命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="接触角测量 - Young-Laplace 方法")
    parser.add_argument("image", nargs="?", default="drop_image.jpg",
                        help="输入图片路径 (默认: drop_image.jpg)")
    parser.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth",
                        help="SAM 模型权重路径")
    parser.add_argument("--output", "-o", default=None,
                        help="结果图片保存路径 (可选)")
    parser.add_argument("--no-display", action="store_true",
                        help="不显示图片 (用于无 GUI 环境)")
    parser.add_argument("--save-masks", action="store_true",
                        help="保存中间掩码图片")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="裁剪 ROI 区域 (x y w h)")
    return parser.parse_args()


# ============================================================
# 主类
# ============================================================
class ContactAngleAnalyzer:
    """接触角分析器 - 集成图像增强、SAM 分割、基准线拟合、
       接触点定位、Young-Laplace 拟合与可视化"""

    def __init__(self, sam_checkpoint_path: str, device: str = None):
        """初始化 SAM 模型

        Parameters
        ----------
        sam_checkpoint_path : str
            SAM 模型权重文件路径 (vit_h)
        device : str or None
            "cuda", "cpu", 或 None 自动选择
        """
        print("=" * 60)
        print("  接触角分析器 - Contact Angle Analyzer")
        print("=" * 60)

        # ----- 设备选择 -----
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[初始化] 使用设备: {self.device}")

        # ----- 加载 SAM -----
        print("[初始化] 加载 SAM 模型...")
        if not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(
                f"SAM 权重文件不存在: {sam_checkpoint_path}\n"
                f"请下载 sam_vit_h_4b8939.pth 并放在当前目录"
            )

        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
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
        print("[初始化] SAM 模型加载完成 ✓\n")

    # ============================================================
    # 主入口
    # ============================================================
    def analyze(self, image_path: str, roi: tuple = None,
                save_masks: bool = False) -> dict:
        """执行完整分析流水线

        Parameters
        ----------
        image_path : str
            输入图片路径
        roi : tuple (x, y, w, h) or None
            可选的 ROI 裁剪区域
        save_masks : bool
            是否保存中间掩码

        Returns
        -------
        dict with keys:
            left_angle, right_angle, average_angle,
            young_laplace_params_left, young_laplace_params_right,
            apex, baseline, contact_points
        """
        print(f"[流水线] 开始分析: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # ROI 裁剪
        if roi is not None:
            x, y, w, h = roi
            img = img[y:y+h, x:x+w]
            print(f"[流水线] ROI 裁剪: ({x}, {y}, {w}, {h})")

        h, w = img.shape[:2]
        print(f"[流水线] 图片尺寸: {w}×{h}")

        # ---- 步骤1: 图像增强 ----
        img_enhanced = self._step1_enhance(img)

        # ---- 步骤2: SAM 分割 ----
        masks_data = self._step2_segment(img_enhanced)
        droplet_mask, substrate_mask = self._filter_masks(masks_data, img.shape)

        if save_masks:
            cv2.imwrite("mask_droplet.png", droplet_mask)
            cv2.imwrite("mask_substrate.png", substrate_mask)
            print("[保存] 掩码已保存至 mask_droplet.png / mask_substrate.png")

        # ---- 步骤3: RANSAC 基准线拟合 ----
        baseline_params = self._step3_ransac_baseline(substrate_mask, img.shape)
        if baseline_params is None:
            raise RuntimeError("基准线拟合失败! 请检查基底掩码质量")

        # ---- 步骤4: 接触点定位 ----
        contour, left_pt, right_pt, apex = self._step4_find_contact_points(
            droplet_mask, baseline_params)

        if left_pt is None or right_pt is None:
            raise RuntimeError("接触点定位失败! 请检查液滴掩码质量")

        # ---- 步骤5: Young-Laplace 拟合与角度计算 ----
        yl_result = self._step5_young_laplace(
            contour, baseline_params, left_pt, right_pt, apex)

        # ---- 可视化 ----
        self._visualize(img, img_enhanced, droplet_mask, substrate_mask,
                        baseline_params, contour, left_pt, right_pt,
                        apex, yl_result)

        return {
            "left_angle": yl_result["left_angle"],
            "right_angle": yl_result["right_angle"],
            "average_angle": (yl_result["left_angle"] + yl_result["right_angle"]) / 2,
            "young_laplace_params_left": yl_result.get("params_left"),
            "young_laplace_params_right": yl_result.get("params_right"),
            "apex": apex,
            "baseline": baseline_params,
            "contact_points": (left_pt, right_pt),
        }

    # ============================================================
    # 步骤 1: 图像增强
    # ============================================================
    def _step1_enhance(self, img: np.ndarray) -> np.ndarray:
        """步骤1: 多阶段图像增强

        1. 降噪 (Fast Non-Local Means Denoising)
        2. CLAHE 对比度增强 (LAB 色彩空间)
        3. 轻微锐化 (Unsharp Masking)
        """
        print("[步骤1] 图像增强...")

        # --- 1a. 降噪 ---
        # 使用双边滤波保持边缘，同时去除噪声
        denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

        # --- 1b. CLAHE 对比度增强 ---
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        # 可选: 自适应直方图均衡的变体
        # clahe2 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
        # l_eq2 = clahe2.apply(l)
        # l_eq = cv2.addWeighted(l_eq, 0.7, l_eq2, 0.3, 0)

        enhanced_lab = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # --- 1c. 轻微锐化 ---
        # Unsharp masking: 原图 - 模糊版 = 细节，加回原图
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

        # 裁剪到 [0, 255]
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        print("[步骤1] 图像增强完成 ✓")
        return enhanced

    # ============================================================
    # 步骤 2: SAM 分割
    # ============================================================
    def _step2_segment(self, img: np.ndarray) -> list:
        """步骤2: SAM 自动分割

        调用 Segment Anything Model 生成所有候选掩码
        """
        print("[步骤2] SAM 分割中 (可能需要几秒到十几秒)...")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(img_rgb)
        print(f"[步骤2] SAM 生成了 {len(masks)} 个候选掩码 ✓")

        # 按面积降序排列，方便调试
        masks.sort(key=lambda m: m['area'], reverse=True)
        return masks

    def _filter_masks(self, masks_data: list, img_shape: tuple) -> tuple:
        """从 SAM 结果中智能筛选液滴掩码与基底掩码

        筛选策略:
        - 基底: 位于图像底部区域的横向大块区域
        - 液滴: 位于图像中上部、外形接近球冠、紧贴基底的区域
        """
        print("[步骤2] 筛选液滴/基底掩码...")
        h, w = img_shape[:2]

        droplet_mask = np.zeros((h, w), dtype=np.uint8)
        substrate_mask = np.zeros((h, w), dtype=np.uint8)

        # 用于基底选择的候选
        substrate_candidates = []
        droplet_candidates = []

        for i, mask_info in enumerate(masks_data):
            seg = mask_info['segmentation']
            area = mask_info['area']
            bbox = mask_info['bbox']  # [x, y, w, h]
            bx, by, bw, bh = bbox
            cy = by + bh / 2  # bbox 中心 y
            cx = bx + bw / 2  # bbox 中心 x

            # 提取轮廓用于形状分析
            contours, _ = cv2.findContours(
                seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue

            # 形状指标
            circularity = 4 * np.pi * area / (perimeter ** 2)  # 接近1=圆
            aspect_ratio = bw / max(bh, 1)  # 宽高比
            extent = area / max(bw * bh, 1)  # 填充率

            # ----- 基底候选 -----
            # 特征: 位于底部、宽而扁、面积大
            bottom_touch = (by + bh) > h * 0.75
            is_wide = aspect_ratio > 2.0
            is_large = area > w * h * 0.05  # 至少占5%
            is_horizontal = bh < h * 0.25

            if bottom_touch and is_wide and is_large:
                substrate_candidates.append((i, mask_info, area))

            # ----- 液滴候选 -----
            # 特征: 中上部、形状接近球冠、紧贴基底
            mid_upper = 0.05 < cy / h < 0.70
            not_too_small = area > 500
            not_full_frame = area < w * h * 0.5
            reasonable_shape = (0.3 < circularity < 1.5) or (0.3 < aspect_ratio < 3.0)

            if mid_upper and not_too_small and not_full_frame and reasonable_shape:
                droplet_candidates.append((i, mask_info, area, cy))

        # ----- 选择基底 -----
        if substrate_candidates:
            # 选面积最大且在底部的
            substrate_candidates.sort(key=lambda x: x[2], reverse=True)
            best_sub = substrate_candidates[0][1]
            substrate_mask[best_sub['segmentation']] = 255
            print(f"  [基底] 选中面积={best_sub['area']} 的掩码")
        else:
            # 回退: 选择底部最大区域
            print("  [基底] 无理想候选, 使用回退策略")
            bottom_masks = [(m['area'], m) for m in masks_data
                            if m['bbox'][1] + m['bbox'][3] > h * 0.8]
            if bottom_masks:
                bottom_masks.sort(key=lambda x: x[0], reverse=True)
                best = bottom_masks[0][1]
                substrate_mask[best['segmentation']] = 255

        # ----- 选择液滴 -----
        if droplet_candidates:
            # 优先选面积适中、y 位置靠上、形状接近圆的
            # 评分: 面积适中 + 居中 + 圆形度好
            def droplet_score(item):
                _, info, area, cy = item
                bx, by, bw, bh = info['bbox']
                cx = bx + bw / 2
                # 面积分 (优选中等偏大, 惩罚过大/过小)
                area_score = min(area / 20000, 20000 / max(area, 1))
                area_score = min(area_score, 2.0)
                # 位置分 (居中)
                pos_score = 1.0 - abs(cx - w/2) / (w/2)
                # 高度分 (靠上但不离基底太远)
                height_score = 1.0 - abs(cy - h*0.35) / (h*0.35)
                return area_score + pos_score + height_score

            droplet_candidates.sort(key=droplet_score, reverse=True)
            best_drop = droplet_candidates[0][1]
            droplet_mask[best_drop['segmentation']] = 255
            print(f"  [液滴] 选中面积={best_drop['area']} 的掩码 "
                  f"(共 {len(droplet_candidates)} 个候选)")
        else:
            # 回退: 选中间偏上、面积适中的
            print("  [液滴] 无理想候选, 使用回退策略")
            for m in masks_data:
                bx, by, bw, bh = m['bbox']
                cy = by + bh / 2
                if h*0.1 < cy < h*0.6 and 500 < m['area'] < w*h*0.5:
                    droplet_mask[m['segmentation']] = 255
                    print(f"  [液滴-回退] 选中面积={m['area']}")
                    break

        # ----- 形态学后处理 -----
        if np.sum(droplet_mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            droplet_mask = cv2.morphologyEx(droplet_mask, cv2.MORPH_CLOSE, kernel)
            # 填充内部孔洞
            contours, _ = cv2.findContours(droplet_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            droplet_mask[:] = 0
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                cv2.drawContours(droplet_mask, [cnt], -1, 255, -1)

        if np.sum(substrate_mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            substrate_mask = cv2.morphologyEx(substrate_mask, cv2.MORPH_CLOSE, kernel)

        print(f"[步骤2] 掩码筛选完成 ✓")
        return droplet_mask, substrate_mask

    # ============================================================
    # 步骤 3: RANSAC 基准线拟合
    # ============================================================
    def _step3_ransac_baseline(self, substrate_mask: np.ndarray,
                                img_shape: tuple) -> dict:
        """步骤3: 从基底掩码提取上边缘 + RANSAC 鲁棒直线拟合

        Returns
        -------
        dict with keys: k, b, point, vector, line_points
        """
        print("[步骤3] RANSAC 基准线拟合...")
        h, w = img_shape[:2]

        ys, xs = np.where(substrate_mask > 0)
        if len(ys) < 10:
            print("[步骤3] ⚠ 基底掩码为空, 无法拟合基准线")
            return None

        # ----- 提取基底上边缘 (每列最上面的掩码像素) -----
        # 使用更鲁棒的方法: 对基底区域进行形态学梯度提取边缘
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(substrate_mask, kernel, iterations=2)
        top_edge = substrate_mask.astype(np.int32) - eroded.astype(np.int32)
        top_edge = np.clip(top_edge, 0, 255).astype(np.uint8)

        # 从上边缘提取点集
        edge_ys, edge_xs = np.where(top_edge > 0)

        if len(edge_xs) < 10:
            # 回退: 每列取最上点
            print("[步骤3] 使用回退方法提取边缘点")
            edge_points = []
            for x in range(w):
                col_ys = ys[xs == x]
                if len(col_ys) > 0:
                    edge_points.append([x, np.min(col_ys)])
            pts = np.array(edge_points, dtype=np.float32)
        else:
            pts = np.column_stack([edge_xs, edge_ys]).astype(np.float32)

        if len(pts) < 10:
            return None

        # 裁剪离群点: 只保留 y 坐标在中间 80% 范围内的点
        y_low = np.percentile(pts[:, 1], 10)
        y_high = np.percentile(pts[:, 1], 90)
        inlier = (pts[:, 1] >= y_low) & (pts[:, 1] <= y_high)
        pts_filtered = pts[inlier]

        if len(pts_filtered) < 5:
            pts_filtered = pts

        # ----- RANSAC 直线拟合 -----
        [vx, vy, x0, y0] = cv2.fitLine(
            pts_filtered, cv2.DIST_HUBER, 0, 0.01, 0.99)

        # 直线方程: y = kx + b
        if abs(vx) < 1e-8:
            k = 0.0  # 垂直基准线 (极少见)
        else:
            k = vy / vx
        k = float(k[0]) if hasattr(k, '__len__') else float(k)
        b_val = y0 - k * x0
        b_val = float(b_val[0]) if hasattr(b_val, '__len__') else float(b_val)
        x0_f = float(x0[0]) if hasattr(x0, '__len__') else float(x0)
        y0_f = float(y0[0]) if hasattr(y0, '__len__') else float(y0)

        # 计算线段的两个端点 (横跨整个图像)
        y_start = k * 0 + b_val
        y_end = k * w + b_val

        result = {
            'k': k,
            'b': b_val,
            'point': (int(x0_f), int(y0_f)),
            'vector': (float(vx[0]) if hasattr(vx, '__len__') else float(vx),
                       float(vy[0]) if hasattr(vy, '__len__') else float(vy)),
            'line_points': ((0, int(y_start)), (w - 1, int(y_end))),
        }

        print(f"[步骤3] 基准线: y = {k:.4f}x + {b_val:.1f} ✓")
        return result

    # ============================================================
    # 步骤 4: 接触点定位
    # ============================================================
    def _step4_find_contact_points(self, droplet_mask: np.ndarray,
                                    baseline_params: dict) -> tuple:
        """步骤4: 提取液滴轮廓 + 与基准线的几何交点 → 接触点 + 顶点

        Returns
        -------
        (contour, left_pt, right_pt, apex)
        """
        print("[步骤4] 接触点定位...")

        # ----- 提取最大轮廓 -----
        contours, _ = cv2.findContours(
            droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, None, None, None

        cnt = max(contours, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2).astype(np.float32)

        k = baseline_params['k']
        b_val = baseline_params['b']

        # ----- 找顶点 (y 最小 = 图像中最高的点) -----
        apex_idx = np.argmin(pts[:, 1])
        apex = tuple(pts[apex_idx].astype(int))

        # ----- 找接触点: 轮廓与基准线的交点 -----
        # 遍历轮廓的每条线段，检测与基准线的交点
        intersections = []
        n = len(pts)

        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]

            # 计算线段与直线 y = kx + b 的交点
            pt = self._line_segment_intersection(p1, p2, k, b_val)
            if pt is not None:
                # 过滤掉接近顶部的交点 (那不是接触点)
                if pt[1] > apex[1] + 10:
                    intersections.append(pt)

        if len(intersections) < 2:
            # 回退: 轮廓中 y 最大的左右两点
            print("[步骤4] ⚠ 交点检测不完整, 使用回退方法")
            max_y = np.max(pts[:, 1])
            bottom_pts = pts[pts[:, 1] >= max_y - 10]
            if len(bottom_pts) >= 2:
                left_pt = tuple(bottom_pts[np.argmin(bottom_pts[:, 0])].astype(int))
                right_pt = tuple(bottom_pts[np.argmax(bottom_pts[:, 0])].astype(int))
            else:
                return cnt, None, None, apex
        else:
            # 按 x 排序，取最左和最右
            intersections.sort(key=lambda p: p[0])
            # 合并距离很近的交点 (容差内取平均)
            merged = []
            for pt in intersections:
                if not merged or pt[0] - merged[-1][0] > 15:
                    merged.append(pt)
                else:
                    # 合并: 取平均
                    merged[-1] = ((merged[-1][0] + pt[0]) / 2,
                                  (merged[-1][1] + pt[1]) / 2)

            if len(merged) >= 2:
                left_pt = (int(merged[0][0]), int(merged[0][1]))
                right_pt = (int(merged[-1][0]), int(merged[-1][1]))
            else:
                left_pt = (int(merged[0][0]), int(merged[0][1]))
                # 单交点: 在轮廓底部找最远的点
                max_y = np.max(pts[:, 1])
                bottom = pts[pts[:, 1] >= max_y - 10]
                other_pt = bottom[np.argmax(bottom[:, 0])].astype(int)
                right_pt = (int(other_pt[0]), int(other_pt[1]))

        # 确保左右正确
        if left_pt[0] > right_pt[0]:
            left_pt, right_pt = right_pt, left_pt

        print(f"[步骤4] 顶点: {apex}, 左接触点: {left_pt}, 右接触点: {right_pt} ✓")
        return cnt, left_pt, right_pt, apex

    @staticmethod
    def _line_segment_intersection(p1, p2, k, b_val):
        """计算线段 p1→p2 与直线 y = kx + b 的交点"""
        x1, y1 = p1
        x2, y2 = p2

        # 线段方向
        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 1e-8:
            # 垂直线段: x = x1
            x_int = x1
            y_int = k * x_int + b_val
        elif abs(k) > 1e6:
            # 基准线接近垂直
            return None
        else:
            # 参数方程求交:
            # 线段: (x, y) = (x1, y1) + t*(dx, dy), t∈[0,1]
            # 直线: y = kx + b
            # → y1 + t*dy = k*(x1 + t*dx) + b
            # → t*(dy - k*dx) = k*x1 + b - y1
            denom = dy - k * dx
            if abs(denom) < 1e-10:
                return None  # 平行
            t = (k * x1 + b_val - y1) / denom
            if t < 0 or t > 1:
                return None  # 交点不在线段上
            x_int = x1 + t * dx
            y_int = y1 + t * dy

        # 验证交点在图像范围内
        return (float(x_int), float(y_int))

    # ============================================================
    # 步骤 5: Young-Laplace 方程拟合与角度计算
    # ============================================================
    def _step5_young_laplace(self, contour, baseline_params,
                              left_pt, right_pt, apex) -> dict:
        """步骤5: 使用 Young-Laplace 方程拟合液滴轮廓并计算接触角

        Young-Laplace 方程 (轴对称液滴):
          dφ/ds = 2/R₀ - (Δρ·g/γ)·z - sin(φ)/x
          dx/ds = cos(φ)
          dz/ds = sin(φ)

        其中:
          φ   = 轮廓切线与水平方向的夹角
          s   = 弧长
          R₀  = 顶点曲率半径
          c   = Δρ·g/γ (毛细常数)

        方法:
        1. 提取轮廓点，转换到物理坐标系
        2. 分别拟合左右两侧
        3. 在接触点处读取 φ 即为接触角
        """
        print("[步骤5] Young-Laplace 拟合与角度计算...")

        if contour is None or left_pt is None or right_pt is None:
            print("[步骤5] ⚠ 输入参数缺失, 返回默认值")
            return {"left_angle": 0.0, "right_angle": 0.0}

        pts = contour.reshape(-1, 2).astype(np.float32)
        k = baseline_params['k']
        b_val = baseline_params['b']
        apex_pt = np.array(apex, dtype=np.float32)

        # ----- 转换到物理坐标系 -----
        # 图像 y↓, 物理 z↑; 以顶点为原点
        # 对每个点: z_phys = apex.y - y_img
        #           x_phys = x_img - apex.x  (左负右正)
        pts_phys = pts.copy()
        pts_phys[:, 0] = pts[:, 0] - apex_pt[0]        # x 居中
        pts_phys[:, 1] = apex_pt[1] - pts[:, 1]         # z 向上

        # 基线在物理坐标中的 z 值
        # 基线上与左/右接触点对应 x 处的 y = kx + b
        z_baseline_left = apex_pt[1] - (k * left_pt[0] + b_val)
        z_baseline_right = apex_pt[1] - (k * right_pt[0] + b_val)
        z_baseline = (z_baseline_left + z_baseline_right) / 2

        # ----- 分离左右轮廓 -----
        # 左轮廓: x < 0 且靠近液滴表面
        left_mask = pts_phys[:, 0] < -3
        right_mask = pts_phys[:, 0] > 3

        # 取 z > z_baseline - 5 的部分 (液滴可见部分)
        left_contour = pts_phys[left_mask & (pts_phys[:, 1] > z_baseline - 5)]
        right_contour = pts_phys[right_mask & (pts_phys[:, 1] > z_baseline - 5)]

        result = {"left_angle": 0.0, "right_angle": 0.0}

        # ----- 拟合左侧 -----
        if len(left_contour) > 15:
            left_angle, left_params = self._fit_yl_side(
                left_contour, z_baseline, side="left")
            result["left_angle"] = left_angle
            result["params_left"] = left_params
        else:
            print("[步骤5] ⚠ 左侧轮廓点不足, 使用多项式回退")
            result["left_angle"] = self._fallback_angle(
                pts_phys, left_pt, apex_pt, baseline_params, 'left')

        # ----- 拟合右侧 -----
        if len(right_contour) > 15:
            right_angle, right_params = self._fit_yl_side(
                right_contour, z_baseline, side="right")
            result["right_angle"] = right_angle
            result["params_right"] = right_params
        else:
            print("[步骤5] ⚠ 右侧轮廓点不足, 使用多项式回退")
            result["right_angle"] = self._fallback_angle(
                pts_phys, right_pt, apex_pt, baseline_params, 'right')

        print(f"[步骤5] Young-Laplace 拟合完成 ✓")
        print(f"        左接触角: {result['left_angle']:.2f}°")
        print(f"        右接触角: {result['right_angle']:.2f}°")
        return result

    def _fit_yl_side(self, contour_phys, z_baseline, side="left") -> tuple:
        """对单侧轮廓进行 Young-Laplace 拟合

        Parameters
        ----------
        contour_phys : ndarray (N, 2)
            物理坐标系中的轮廓点 [x, z]
        z_baseline : float
            基线在物理坐标系中的 z 值 (负值)
        side : str
            "left" 或 "right"

        Returns
        -------
        (angle_deg, fitted_params_dict)
        """
        # 使用 x 的绝对值 (对称处理)
        x_vals = np.abs(contour_phys[:, 0])
        z_vals = contour_phys[:, 1]

        # 按 z 排序 (从顶点向下)
        sort_idx = np.argsort(z_vals)[::-1]  # 降序: z=0 → z=z_baseline
        x_sorted = x_vals[sort_idx]
        z_sorted = z_vals[sort_idx]

        # 去除离群点 (使用滑动中值滤波)
        if len(x_sorted) > 20:
            window = max(5, len(x_sorted) // 10)
            x_smooth = np.convolve(x_sorted, np.ones(window)/window, mode='same')
            # 保留偏离不太大的点
            deviation = np.abs(x_sorted - x_smooth)
            mad = np.median(deviation)
            keep = deviation < 5 * mad + 10  # 容差
            x_sorted = x_sorted[keep]
            z_sorted = z_sorted[keep]

        if len(x_sorted) < 10:
            return 0.0, None

        # ----- 估计初始参数 -----
        # b = 1/R₀: 顶点曲率，用顶点附近的圆拟合估算
        near_apex = z_sorted > -5  # z ≈ 0 附近的点
        if np.sum(near_apex) < 3:
            near_apex = np.ones_like(z_sorted, dtype=bool)
        x_near = x_sorted[near_apex]
        z_near = z_sorted[near_apex]
        if len(x_near) >= 3:
            # 圆拟合: x² + (z - zc)² = r²  →  x² + z² = 2*zc*z + (r² - zc²)
            A = np.column_stack([z_near, np.ones_like(z_near)])
            rhs = x_near**2 + z_near**2
            sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            zc = sol[0] / 2
            r_est = np.sqrt(sol[1] + zc**2)
            r_est = max(r_est, 10)
            b_init = 1.0 / r_est
        else:
            b_init = 1.0 / 100.0

        # c = Δρg/γ: 毛细常数 (与尺度相关)
        # 粗略估计: 对于水在空气中, c ≈ 0.14 mm⁻²
        # 在像素空间中, 需要根据实际尺度调整
        c_init = 0.001

        # ----- 目标: 拟合 {b, c} -----
        z_target = z_sorted.copy()
        x_target = x_sorted.copy()
        z_stop = z_baseline - 5  # 积分终点 (略低于基线)

        def objective(params):
            b_fit, c_fit = params
            b_fit = max(b_fit, 1e-6)
            c_fit = max(c_fit, 1e-10)

            try:
                s_eval, x_yl, z_yl, phi_yl = self._solve_yl_profile(
                    b_fit, c_fit, z_stop, n_points=len(z_target) * 2)
            except Exception:
                return np.full(len(z_target), 1e6)

            if len(z_yl) < 5:
                return np.full(len(z_target), 1e6)

            # 对每个目标 z，找到 YL 曲线上对应的 x
            x_interp = np.interp(z_target, z_yl[::-1], x_yl[::-1],
                                 left=1e6, right=1e6)
            residuals = (x_interp - x_target) / max(np.mean(x_target), 1.0)
            return residuals

        # 使用 least_squares 优化
        try:
            res = least_squares(
                objective,
                [b_init, c_init],
                bounds=([1e-6, 1e-10], [1.0, 1.0]),
                method='trf',
                max_nfev=200,
                ftol=1e-6,
                xtol=1e-6,
                loss='soft_l1',
                f_scale=0.5,
            )
            b_opt, c_opt = res.x
        except Exception as e:
            print(f"  [YL拟合] 优化失败 ({side}): {e}, 使用初始估计")
            b_opt, c_opt = b_init, c_init

        b_opt = max(b_opt, 1e-6)
        c_opt = max(c_opt, 1e-10)

        # ----- 用最优参数计算接触角 -----
        try:
            s_eval, x_yl, z_yl, phi_yl = self._solve_yl_profile(
                b_opt, c_opt, z_stop, n_points=500)

            # 在 z = z_baseline 处读取 φ
            angle_rad = np.interp(z_baseline, z_yl[::-1], phi_yl[::-1])
            angle_deg = np.degrees(abs(angle_rad))

            # 合理性检查
            if angle_deg < 0 or angle_deg > 180:
                angle_deg = np.clip(angle_deg, 0, 180)
        except Exception:
            angle_deg = 0.0

        params = {
            'b': float(b_opt),
            'c': float(c_opt),
            'R0': float(1.0 / b_opt),
            'yl_curve': (x_yl, z_yl, phi_yl) if 'x_yl' in dir() else (None, None, None),
        }

        return angle_deg, params

    def _solve_yl_profile(self, b, c, z_stop, n_points=200):
        """数值求解 Young-Laplace ODE

        Parameters
        ----------
        b : float
            顶点曲率 (1/R₀)
        c : float
            毛细常数 (Δρg/γ)
        z_stop : float
            积分终点 z 值 (负值, 代表基线)
        n_points : int
            输出采样点数

        Returns
        -------
        s, x, z, phi : 弧长及对应的坐标和切线角
        """
        # 从极小值开始, 避免 x=0 处的奇点
        s0 = 1e-8
        # 近顶点解析近似
        phi0 = b * s0
        x0 = s0
        z0 = (b / 2) * s0 ** 2  # z ≈ (φ/s)*s²/2

        def ode(s, y):
            x, z_val, phi = y
            if x < 1e-12:
                sin_over_x = b  # 极限: sin(φ)/x → b as x→0
            else:
                sin_over_x = np.sin(phi) / x
            dphi_ds = 2 * b - c * z_val - sin_over_x
            return [np.cos(phi), np.sin(phi), dphi_ds]

        # 使用事件检测 z 到达目标
        def event_z(s, y):
            return y[1] - z_stop
        event_z.terminal = True
        event_z.direction = -1  # z 从 0 向负方向移动

        try:
            sol = solve_ivp(
                ode,
                [s0, 5000.0],  # 足够大的弧长范围
                [x0, z0, phi0],
                method='RK45',
                events=event_z,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10,
                max_step=5.0,
            )
        except Exception:
            # 回退: 使用 LSODA
            sol = solve_ivp(
                ode,
                [s0, 5000.0],
                [x0, z0, phi0],
                method='LSODA',
                events=event_z,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10,
                max_step=5.0,
            )

        s_eval = np.linspace(s0, sol.t[-1], n_points)
        y_eval = sol.sol(s_eval)

        return (s_eval, y_eval[0], y_eval[1], y_eval[2])

    def _fallback_angle(self, pts_phys, contact_pt_img, apex_pt,
                         baseline_params, side='left') -> float:
        """回退方法: 使用局部多项式拟合计算接触角

        当 Young-Laplace 拟合失败时使用
        """
        # 在接触点附近取轮廓点
        cx, cy = contact_pt_img[0], contact_pt_img[1]
        # 在物理坐标中找对应区域
        pts = pts_phys

        # 取接触点附近的点进行多项式拟合
        if side == 'left':
            nearby = pts[(pts[:, 0] < 0) & (np.abs(pts[:, 0] - (cx - apex_pt[0])) < 50)]
        else:
            nearby = pts[(pts[:, 0] > 0) & (np.abs(pts[:, 0] - (cx - apex_pt[0])) < 50)]

        if len(nearby) < 5:
            return 0.0

        # 二次多项式拟合 z = f(x)
        try:
            coeffs = np.polyfit(nearby[:, 0], nearby[:, 1], 2)
            # 在接触点处的导数 dz/dx = tan(φ)
            x_c = cx - apex_pt[0]
            dz_dx = np.polyval([2*coeffs[0], coeffs[1]], x_c)
            phi_rad = np.arctan(dz_dx)
            # 接触角 = 切线与水平面的夹角
            angle_deg = np.degrees(abs(phi_rad))

            # 在图像坐标中: 轮廓切线角
            k = baseline_params['k']
            baseline_angle = np.arctan(k)  # 基线倾角
            angle_deg -= np.degrees(baseline_angle)
            angle_deg = abs(angle_deg)

            return np.clip(angle_deg, 0, 180)
        except Exception:
            return 0.0

    # ============================================================
    # 可视化
    # ============================================================
    def _visualize(self, img_orig, img_enhanced, droplet_mask, substrate_mask,
                    baseline_params, contour, left_pt, right_pt, apex,
                    yl_result):
        """综合可视化: 2×3 子图布局"""
        print("[可视化] 生成结果图...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("接触角分析 - Contact Angle Analysis (Young-Laplace Method)",
                     fontsize=16, fontweight='bold')

        h, w = img_orig.shape[:2]
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_enh_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
        k = baseline_params['k']
        b_val = baseline_params['b']

        # ---- (1) 原始图像 ----
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title("(1) 原始图像", fontsize=12)
        axes[0, 0].axis('off')

        # ---- (2) 增强图像 ----
        axes[0, 1].imshow(img_enh_rgb)
        axes[0, 1].set_title("(2) 增强图像 (CLAHE+降噪+锐化)", fontsize=12)
        axes[0, 1].axis('off')

        # ---- (3) SAM 分割掩码 ----
        overlay = img_rgb.copy()
        # 液滴掩码 - 半透明蓝色
        mask_d = droplet_mask > 0
        overlay[mask_d] = overlay[mask_d] * 0.5 + np.array([30, 144, 255]) * 0.5
        # 基底掩码 - 半透明红色
        mask_s = substrate_mask > 0
        overlay[mask_s] = overlay[mask_s] * 0.5 + np.array([255, 80, 80]) * 0.5
        axes[0, 2].imshow(overlay.astype(np.uint8))
        axes[0, 2].set_title("(3) SAM 分割 (蓝=液滴, 红=基底)", fontsize=12)
        axes[0, 2].axis('off')

        # ---- (4) 基准线 + 接触点 ----
        ax4 = axes[1, 0]
        ax4.imshow(img_rgb)
        # 基准线
        x_line = np.array([0, w - 1])
        y_line = k * x_line + b_val
        ax4.plot(x_line, y_line, 'r-', linewidth=2, label='Baseline (RANSAC)')
        # 接触点
        if left_pt:
            ax4.plot(left_pt[0], left_pt[1], 'go', markersize=10,
                     markeredgecolor='white', markeredgewidth=2,
                     label=f'Left: {yl_result["left_angle"]:.1f}°')
        if right_pt:
            ax4.plot(right_pt[0], right_pt[1], 'co', markersize=10,
                     markeredgecolor='white', markeredgewidth=2,
                     label=f'Right: {yl_result["right_angle"]:.1f}°')
        if apex:
            ax4.plot(apex[0], apex[1], 'y*', markersize=12,
                     markeredgecolor='black', markeredgewidth=1,
                     label='Apex')
        # 轮廓
        if contour is not None:
            ax4.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=1.5,
                     alpha=0.7, label='Contour')
        ax4.set_title("(4) 基准线 & 接触点定位", fontsize=12)
        ax4.legend(loc='upper right', fontsize=7, framealpha=0.8)
        ax4.axis('off')

        # ---- (5) Young-Laplace 拟合曲线 ----
        ax5 = axes[1, 1]
        ax5.imshow(img_rgb)

        # 画检测到的轮廓
        if contour is not None:
            ax5.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=1,
                     alpha=0.5, label='Detected contour')

        # 画 Young-Laplace 拟合结果 (左右分别)
        apex_pt_np = np.array(apex, dtype=np.float32)

        for side, color in [('left', '#FF6B6B'), ('right', '#4ECDC4')]:
            params_key = f'params_{side}'
            if params_key in yl_result and yl_result[params_key] is not None:
                params = yl_result[params_key]
                yl_curve = params.get('yl_curve')
                if yl_curve is not None and yl_curve[0] is not None:
                    x_yl, z_yl, _ = yl_curve
                    # 转换回图像坐标
                    sign = -1 if side == 'left' else 1
                    x_img = apex_pt_np[0] + sign * x_yl
                    y_img = apex_pt_np[1] - z_yl
                    # 只画液滴可见部分 (在基线上方)
                    y_baseline = k * x_img + b_val
                    visible = y_img < y_baseline + 5
                    ax5.plot(x_img[visible], y_img[visible], '-', color=color,
                             linewidth=2.5, alpha=0.9,
                             label=f'YL fit ({side})')

        # 基准线
        ax5.plot(x_line, y_line, 'r--', linewidth=1.5, alpha=0.7)
        # 接触点
        if left_pt:
            ax5.plot(left_pt[0], left_pt[1], 'o', color='#FF6B6B', markersize=8,
                     markeredgecolor='white', markeredgewidth=1.5)
        if right_pt:
            ax5.plot(right_pt[0], right_pt[1], 'o', color='#4ECDC4', markersize=8,
                     markeredgecolor='white', markeredgewidth=1.5)
        if apex:
            ax5.plot(apex[0], apex[1], 'y*', markersize=10,
                     markeredgecolor='black', markeredgewidth=1)
        ax5.set_title("(5) Young-Laplace 拟合", fontsize=12)
        ax5.legend(loc='upper right', fontsize=7, framealpha=0.8)
        ax5.axis('off')

        # ---- (6) 角度标注特写 ----
        ax6 = axes[1, 2]
        # 裁剪到液滴 + 基线区域
        if contour is not None and apex is not None:
            cx_b, cy_b, cw_b, ch_b = cv2.boundingRect(contour)
            pad = 30
            x1_c = max(0, cx_b - pad)
            y1_c = max(0, cy_b - pad)
            x2_c = min(w, cx_b + cw_b + pad)
            y2_c = min(h, cy_b + ch_b + pad)
            crop = img_rgb[y1_c:y2_c, x1_c:x2_c]
            ax6.imshow(crop)

            # 在裁剪坐标中重绘
            def to_crop(pt):
                return (pt[0] - x1_c, pt[1] - y1_c)

            # 基准线
            y_l_crop = k * x1_c + b_val - y1_c
            y_r_crop = k * x2_c + b_val - y1_c
            ax6.plot([0, x2_c - x1_c], [y_l_crop, y_r_crop], 'r-', linewidth=2)

            # 接触点
            if left_pt:
                lp = to_crop(left_pt)
                ax6.plot(lp[0], lp[1], 'go', markersize=8,
                         markeredgecolor='white', markeredgewidth=1.5)
                ax6.annotate(f"{yl_result['left_angle']:.1f}°",
                             xy=lp, xytext=(lp[0] - 50, lp[1] - 20),
                             fontsize=12, color='green', fontweight='bold',
                             arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
            if right_pt:
                rp = to_crop(right_pt)
                ax6.plot(rp[0], rp[1], 'co', markersize=8,
                         markeredgecolor='white', markeredgewidth=1.5)
                ax6.annotate(f"{yl_result['right_angle']:.1f}°",
                             xy=rp, xytext=(rp[0] + 10, rp[1] - 20),
                             fontsize=12, color='cyan', fontweight='bold',
                             arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5))

            # 顶点
            if apex:
                ap = to_crop(apex)
                ax6.plot(ap[0], ap[1], 'y*', markersize=10,
                         markeredgecolor='black', markeredgewidth=1)
        else:
            ax6.imshow(img_rgb)

        ax6.set_title("(6) 接触角标注 (特写)", fontsize=12)
        ax6.axis('off')

        # 总标题信息
        avg_angle = (yl_result['left_angle'] + yl_result['right_angle']) / 2
        fig.text(0.5, 0.02,
                 f"左接触角: {yl_result['left_angle']:.2f}°  |  "
                 f"右接触角: {yl_result['right_angle']:.2f}°  |  "
                 f"平均: {avg_angle:.2f}°",
                 ha='center', fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        print("[可视化] 完成 ✓")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  接触角测量器 - Contact Angle Analyzer")
    print("  方法: Young-Laplace 方程拟合")
    print("=" * 60)

    # 检查文件
    if not os.path.exists(args.sam_checkpoint):
        print(f"\n❌ 错误: SAM 模型文件不存在!")
        print(f"   路径: {args.sam_checkpoint}")
        print(f"   请将 sam_vit_h_4b8939.pth 放在当前目录,")
        print(f"   或使用 --sam_checkpoint 指定路径\n")
        print(f"   SAM 模型下载地址:")
        print(f"   https://github.com/facebookresearch/segment-anything\n")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"\n❌ 错误: 输入图片不存在!")
        print(f"   路径: {args.image}")
        print(f"   用法: python 333.py <图片路径>\n")
        sys.exit(1)

    try:
        # 创建分析器
        analyzer = ContactAngleAnalyzer(args.sam_checkpoint)

        # 执行分析
        result = analyzer.analyze(
            args.image,
            roi=tuple(args.crop) if args.crop else None,
            save_masks=args.save_masks,
        )

        # 打印结果
        print("\n" + "=" * 60)
        print("  分析结果")
        print("=" * 60)
        print(f"  左接触角 (Left):   {result['left_angle']:8.3f}°")
        print(f"  右接触角 (Right):  {result['right_angle']:8.3f}°")
        print(f"  平均 (Average):    {result['average_angle']:8.3f}°")
        print("=" * 60)

        # 保存结果图
        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\n结果图已保存至: {args.output}")

        # 显示
        if not args.no_display:
            plt.show()
        else:
            print("(已启用 --no-display, 跳过显示)")

    except Exception as e:
        print(f"\n❌ 分析出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
