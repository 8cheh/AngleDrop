import cv2
import numpy as np
import matplotlib.pyplot as plt
def clahe_enhance(image, clip_limit=2.0, grid_size=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.2, gaussian, -0.2, 0)
    return sharpened
def find_droplet_and_enhance(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取 {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #1.圆形定位
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    clahe_temp = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe_temp.apply(blurred)
    
    circles = cv2.HoughCircles(
        enhanced_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=25, minRadius=5, maxRadius=50
    )
    if circles is None:
        print("未找到液滴")
        return
    circles = np.round(circles[0, :]).astype("int")
    cx, cy, r = circles[0]
    print(f"找到液滴：中心({cx}, {cy}), 半径 {r}")
    # 2. 截取 ROI（液滴周围区域）
    margin = r * 2
    x1, y1 = max(0, cx - margin), max(0, cy - margin)
    x2, y2 = min(img.shape[1], cx + margin), min(img.shape[0], cy + margin)
    roi = img[y1:y2, x1:x2]
    # 3. CLAHE 增强
    enhanced_roi = clahe_enhance(roi, clip_limit=2.0, grid_size=8)
    # 4. 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original ROI\nSize: {roi.shape[1]}x{roi.shape[0]}")
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Enhanced ROI (CLAHE + Sharpen)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    # 5. 保存并自动调用
    cv2.imwrite("enhanced_roi.jpg", enhanced_roi)
    print("已保存增强图片到 enhanced_roi.jpg")
    
    try:
        # 导入
        from Segmenter import DropletSegmenter
        seg = DropletSegmenter()
        # 重新定位
        contour, base_y, valid_cnt, roi_info = seg.segment_and_fit_baseline(enhanced_roi, cx, cy, r)
        seg.visualize(enhanced_roi, contour, valid_cnt, base_y)
        print(f"分割完成！基准线 y = {base_y:.1f}")
    except ImportError:
        print("未找到 Segmenter.py，请确保它在同一目录下。")
    except Exception as e:
        print(f"分割过程出错: {e}")
    cv2.imwrite("enhanced_roi.jpg", enhanced_roi)
    print("已保存增强图片，准备进入分割阶段...")
    try:
        # 1. 调用
        from Segmenter import DropletSegmenter
        seg = DropletSegmenter()
        contour, base_y, valid_cnt, roi_info = seg.segment_and_fit_baseline(enhanced_roi, cx, cy, r)
        seg.visualize(enhanced_roi, contour, valid_cnt, base_y)
        print(f"分割完成！基准线 y = {base_y:.1f}")
        # 2. 调用 Young-Laplace 计算器
        from YLCalculator import YLCalculator
        calc = YLCalculator()
        angle = calc.calculate_contact_angle(valid_cnt, base_y, r)
        calc.visualize_fit(valid_cnt, base_y, angle)
        print(f"\n✅ 最终接触角: {angle:.2f}°")
        
    except ImportError as e:
        print(f"模块导入失败: {e}")
    except Exception as e:
        print(f"计算过程出错: {e}")


if __name__ == "__main__":
    # 替换为你的图片路径
    find_droplet_and_enhance("drop_image.jpg")
