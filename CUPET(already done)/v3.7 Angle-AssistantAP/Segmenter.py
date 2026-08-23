import cv2
import numpy as np
import matplotlib.pyplot as plt

class DropletSegmenter:
    def __init__(self):
        pass

    def segment_and_fit_baseline(self, enhanced_roi, cx, cy, r):
        
        h, w = enhanced_roi.shape[:2]
        gray = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. 重新定位
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
            param1=30, param2=15, minRadius=max(3, r//3), maxRadius=r*2
        )
        if circles is not None:
            cx_roi, cy_roi, r_roi = np.round(circles[0, 0]).astype("int")
        else:
            cx_roi, cy_roi, r_roi = w//2, h//2, r

        # 2.临时基准线
        temp_baseline = int(cy_roi + r_roi * 1.2) 
        temp_baseline = min(temp_baseline, h - 5) 
        
        print(f"   临时基准线 y = {temp_baseline}")

        # 3. 强制清除
        gray_clean = gray.copy()
        gray_clean[temp_baseline:, :] = 0  # 涂黑
        
        # 4. Otsu
        _, thresh = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 5. 提取轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("阈值分割未提取到轮廓")
        
        cnt = max(contours, key=cv2.contourArea)
        contour = cnt.reshape(-1, 2)[:, [1, 0]]  # [x,y] -> [y,x]

        # 6.真实基准线
        unique_ys = np.unique(contour[:, 0])
        unique_ys = np.sort(unique_ys)[::-1] # 从大到小（从下往上）
        
        baseline_y = temp_baseline # 默认值
        prev_width = 0
        
        for y in unique_ys:
            xs_at_y = contour[contour[:, 0] == y, 1]
            if len(xs_at_y) > 0:
                width = np.max(xs_at_y) - np.min(xs_at_y)
                if width < w * 0.8 and prev_width >= w * 0.8:
                    baseline_y = y+3
                    break
                prev_width = width
        if baseline_y == temp_baseline:
            for y in unique_ys:
                xs_at_y = contour[contour[:, 0] == y, 1]
                if len(xs_at_y) > 0:
                    width = np.max(xs_at_y) - np.min(xs_at_y)
                    if width < w * 0.7: # 宽度小于 70% 图片宽度
                        baseline_y = y
                        break

        print(f"   真实基准线 y = {baseline_y}")
        valid_mask = contour[:, 0] < baseline_y
        valid_contour = contour[valid_mask]
        
        return contour, baseline_y, valid_contour, (cx_roi, cy_roi, r_roi)
    
    def _fit_baseline(self, gray, cy, r):
        """
        基于行梯度的基准线检测 (寻找水平分界线)
        """
        h, w = gray.shape
        y_start = max(0, int(cy + r * 0.5))
        y_end = min(h, int(cy + r * 2.0))
        
        if y_start >= y_end:
            return cy + r
        roi_search = gray[y_start:y_end, :]
        sobel_y = cv2.Sobel(roi_search, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel = np.absolute(sobel_y)
        row_scores = np.sum(abs_sobel, axis=1)
        best_row_idx = int(np.mean(top_indices))
        baseline_y = best_row_idx + y_start
        if baseline_y <= cy:
            baseline_y = cy + r
            
        print(f"   [梯度法] 基准线 y = {baseline_y:.1f}")
        return baseline_y

    def visualize(self, enhanced_roi, contour, valid_contour, baseline_y):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB))
        axes[0].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5, label='Raw Contour')
        axes[0].axhline(y=baseline_y, color='yellow', linestyle='--', linewidth=2, label='Baseline')
        axes[0].set_title("Raw Segmentation & Baseline")
        axes[0].legend()
        
        axes[1].imshow(cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB))
        if len(valid_contour) > 0:
            axes[1].plot(valid_contour[:, 1], valid_contour[:, 0], 'lime', linewidth=2, label='Valid Contour')
        axes[1].axhline(y=baseline_y, color='red', linewidth=2, label='Baseline')
        axes[1].set_title("Clipped Contour (Ready for Y-L Fit)")
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()



# 测试入口

if __name__ == "__main__":
    # 假设你已经从 Picbetter 拿到了 enhanced_roi
    # 这里为了演示，直接读取你增强后保存的图片（需先取消 Picbetter 里的保存注释）
    import os
    if os.path.exists("enhanced_roi.jpg"):
        roi = cv2.imread("enhanced_roi.jpg")
        seg = DropletSegmenter()
        contour, base_y, valid_cnt, roi_info = seg.segment_and_fit_baseline(roi, 0, 0, 20)
        seg.visualize(roi, contour, valid_cnt, base_y)
        print(f"基准线 y = {base_y:.1f}, 有效点数 = {len(valid_cnt)}")
    else:
        print("请先运行 Picbetter.py 并取消保存注释，生成 enhanced_roi.jpg")
