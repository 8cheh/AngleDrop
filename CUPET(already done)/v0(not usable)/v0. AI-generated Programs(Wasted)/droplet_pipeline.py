import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 读取图片
# =========================
img = cv2.imread("drop_image.jpg")

if img is None:
    raise ValueError("图片路径错误")


# =========================
# 2. matplotlib 手动选ROI
# =========================
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.set_title("Drag ROI (close window after selection)")

print("👉 用鼠标框选区域，然后关闭窗口")

roi = plt.ginput(2)  # 点击两个点（左上 + 右下）
plt.close()

(x1, y1), (x2, y2) = roi
x1, x2 = int(min(x1,x2)), int(max(x1,x2))
y1, y2 = int(min(y1,y2)), int(max(y1,y2))


# =========================
# 3. ROI
# =========================
crop = img[y1:y2, x1:x2]


# =========================
# 4. Hough（稳定增强版）
# =========================
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (7,7), 1.2)

clahe = cv2.createCLAHE(2.5, (8,8))
gray = clahe.apply(gray)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=40,
    param1=80,
    param2=25,
    minRadius=5,
    maxRadius=100
)

if circles is None:
    print("❌ 没检测到液滴")
    exit()

circles = np.round(circles[0]).astype(int)
cx, cy, r = max(circles, key=lambda c: c[2])

cx += x1
cy += y1


# =========================
# 5. 输出
# =========================
out = img.copy()
cv2.circle(out, (cx,cy), r, (0,0,255), 2)

plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.title(f"Droplet detected r={r}")
plt.axis("off")
plt.show()

print("✅ droplet:", cx, cy, r)
