"""
接触角自动测量 Web 应用
上传液滴图片 → 自动输出接触角
"""
import os
import io
import base64
import traceback
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory

# ---------- 复用项目中的核心逻辑 ----------

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


def pil_to_base64(fig):
    """将 matplotlib figure 转为 base64 PNG"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def segment_droplet(enhanced_roi, cx, cy, r):
    """分割液滴，返回轮廓、基准线和可视化"""
    h, w = enhanced_roi.shape[:2]
    gray = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2GRAY)

    # 1. 重新定位
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
        param1=30, param2=15, minRadius=max(3, r // 3), maxRadius=r * 2
    )
    if circles is not None:
        cx_roi, cy_roi, r_roi = np.round(circles[0, 0]).astype("int")
    else:
        cx_roi, cy_roi, r_roi = w // 2, h // 2, r

    # 2. 临时基准线
    temp_baseline = int(cy_roi + r_roi * 1.2)
    temp_baseline = min(temp_baseline, h - 5)

    # 3. 强制清除基准线以下
    gray_clean = gray.copy()
    gray_clean[temp_baseline:, :] = 0

    # 4. Otsu 阈值
    _, thresh = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. 提取轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("阈值分割未提取到轮廓")
    cnt = max(contours, key=cv2.contourArea)
    contour = cnt.reshape(-1, 2)[:, [1, 0]]  # [x,y] -> [y,x]

    # 6. 真实基准线
    unique_ys = np.unique(contour[:, 0])
    unique_ys = np.sort(unique_ys)[::-1]
    baseline_y = temp_baseline
    prev_width = 0
    for y in unique_ys:
        xs_at_y = contour[contour[:, 0] == y, 1]
        if len(xs_at_y) > 0:
            width = np.max(xs_at_y) - np.min(xs_at_y)
            if width < w * 0.8 and prev_width >= w * 0.8:
                baseline_y = y + 3
                break
            prev_width = width
    if baseline_y == temp_baseline:
        for y in unique_ys:
            xs_at_y = contour[contour[:, 0] == y, 1]
            if len(xs_at_y) > 0:
                width = np.max(xs_at_y) - np.min(xs_at_y)
                if width < w * 0.7:
                    baseline_y = y
                    break

    valid_mask = contour[:, 0] < baseline_y
    valid_contour = contour[valid_mask]

    # --- 可视化分割 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB))
    axes[0].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5, label='Raw Contour')
    axes[0].axhline(y=baseline_y, color='yellow', linestyle='--', linewidth=2, label='Baseline')
    axes[0].set_title("Raw Segmentation & Baseline")
    axes[0].legend(fontsize=8)

    axes[1].imshow(cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB))
    if len(valid_contour) > 0:
        axes[1].plot(valid_contour[:, 1], valid_contour[:, 0], 'lime', linewidth=2, label='Valid Contour')
    axes[1].axhline(y=baseline_y, color='red', linewidth=2, label='Baseline')
    axes[1].set_title("Clipped Contour (Ready for Y-L Fit)")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    seg_img = pil_to_base64(fig)
    plt.close(fig)

    return contour, baseline_y, valid_contour, (cx_roi, cy_roi, r_roi), seg_img


def calculate_angle(valid_contour, baseline_y, r_guess):
    """Young-Laplace 拟合计算接触角"""
    from scipy.optimize import least_squares
    from scipy.integrate import solve_ivp

    if len(valid_contour) < 10:
        raise ValueError("有效轮廓点太少")

    contour_y = valid_contour[:, 0]
    contour_x = valid_contour[:, 1]
    contour_z = baseline_y - contour_y

    idx_tip = np.argmax(contour_z)
    x_tip = contour_x[idx_tip]
    z_tip = contour_z[idx_tip]

    sort_idx = np.argsort(contour_x)
    sorted_x = contour_x[sort_idx]
    sorted_z = contour_z[sort_idx]
    dx = np.diff(sorted_x)
    gaps = np.where(dx > 10)[0]

    if len(gaps) > 0:
        segments = []
        start = 0
        for gap in gaps:
            segments.append((start, gap))
            start = gap + 1
        segments.append((start, len(sorted_x) - 1))
        longest_seg = max(segments, key=lambda s: s[1] - s[0])
        sorted_x = sorted_x[longest_seg[0]:longest_seg[1] + 1]
        sorted_z = sorted_z[longest_seg[0]:longest_seg[1] + 1]

    z_threshold = z_tip * 0.15
    valid_mask = sorted_z > z_threshold
    x_clean = sorted_x[valid_mask]
    z_clean = sorted_z[valid_mask]

    idx_tip_new = np.argmax(z_clean)
    x_tip_clean = x_clean[idx_tip_new]

    x_norm = x_clean - x_tip_clean
    z_norm = z_clean - z_tip

    def fit_one_side(x_norm, z_norm, side='right'):
        if side == 'right':
            mask = x_norm > 0
            x_side = x_norm[mask]
            z_side = z_norm[mask]
        else:
            mask = x_norm < 0
            x_side = -x_norm[mask]
            z_side = z_norm[mask]

        if len(x_side) < 5:
            return 90.0

        sort_idx = np.argsort(x_side)
        x_side = x_side[sort_idx]
        z_side = z_side[sort_idx]
        z_phys = -z_side

        def yl_ode(s, y, b):
            x, z, phi = y
            if x < 1e-6:
                return [1.0, 0.0, b]
            return [np.cos(phi), np.sin(phi), b - np.sin(phi) / x]

        def objective(params):
            b = params[0]
            if b <= 0:
                return [1e6]
            y0 = [1e-6, 0, 0.01]
            s_span = (0, np.max(x_side) * 1.5)
            try:
                sol = solve_ivp(yl_ode, s_span, y0, args=(b,), method='RK45', dense_output=True)
                if not sol.success or len(sol.t) < 10:
                    return np.full(len(x_side), 1e6)
                theo_z = sol.sol(x_side)[1]
                return theo_z - z_phys
            except:
                return np.full(len(x_side), 1e6)

        R_guess = (np.max(x_side) + np.max(z_phys)) / 2
        if R_guess < 5:
            R_guess = r_guess
        p0 = [2.0 / R_guess]

        try:
            res = least_squares(objective, p0, bounds=([0], [np.inf]))
            b_opt = res.x[0]
        except:
            return 90.0

        try:
            sol_final = solve_ivp(yl_ode, (0, np.max(x_side)), [1e-6, 0, 0.01], args=(b_opt,), method='RK45')
            phi_at_end = sol_final.y[2, -1]
            angle_deg = np.degrees(phi_at_end)
            if angle_deg < 0:
                angle_deg += 180
            if angle_deg > 180:
                angle_deg = 360 - angle_deg
            return angle_deg
        except:
            return 90.0

    angle_left = fit_one_side(x_norm, z_norm, 'left')
    angle_right = fit_one_side(x_norm, z_norm, 'right')
    angle_avg = (angle_left + angle_right) / 2

    # --- 可视化 Y-L 拟合 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(contour_x, contour_z, 'bo', markersize=3, label='Contour Points')
    ax.plot(x_tip, z_tip, 'r*', markersize=15, label='Tip')
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Baseline (z=0)')
    ax.text(
        x_tip + 10, z_tip - 10,
        f'Contact Angle: {angle_avg:.1f}°',
        color='red', fontsize=14, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='red')
    )
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('z (pixels, upward)')
    ax.set_title('Young-Laplace Fit Result')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    yl_img = pil_to_base64(fig)
    plt.close(fig)

    return {
        'angle_avg': round(float(angle_avg), 2),
        'angle_left': round(float(angle_left), 2),
        'angle_right': round(float(angle_right), 2),
        'yl_image': yl_img
    }


def process_image(image_path):
    """完整处理管线：读取 → 定位 → 增强 → 分割 → Y-L 计算"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 圆形定位
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    clahe_temp = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe_temp.apply(blurred)
    circles = cv2.HoughCircles(
        enhanced_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=25, minRadius=5, maxRadius=50
    )
    if circles is None:
        # 尝试更宽松的参数
        circles = cv2.HoughCircles(
            enhanced_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=30, param2=20, minRadius=5, maxRadius=200
        )
    if circles is None:
        raise ValueError("未找到液滴，请确保图片中有清晰的液滴轮廓")

    circles = np.round(circles[0, :]).astype("int")
    cx, cy, r = circles[0]

    # 2. 截取 ROI
    margin = r * 2
    x1, y1 = max(0, cx - margin), max(0, cy - margin)
    x2, y2 = min(img.shape[1], cx + margin), min(img.shape[0], cy + margin)
    roi = img[y1:y2, x1:x2]

    # 3. CLAHE 增强
    enhanced_roi = clahe_enhance(roi, clip_limit=2.0, grid_size=8)

    # ROI 对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original ROI\nSize: {roi.shape[1]}x{roi.shape[0]}")
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Enhanced ROI (CLAHE + Sharpen)")
    axes[1].axis('off')
    plt.tight_layout()
    roi_img = pil_to_base64(fig)
    plt.close(fig)

    # 4. 分割
    contour, baseline_y, valid_contour, roi_info, seg_img = segment_droplet(enhanced_roi, cx, cy, r)

    # 5. Young-Laplace 计算
    result = calculate_angle(valid_contour, baseline_y, r)

    return {
        'cx': int(cx), 'cy': int(cy), 'radius': int(r),
        'baseline_y': float(baseline_y),
        'contour_points': len(valid_contour),
        'angle_avg': result['angle_avg'],
        'angle_left': result['angle_left'],
        'angle_right': result['angle_right'],
        'roi_image': roi_img,
        'seg_image': seg_img,
        'yl_image': result['yl_image']
    }


# ==================== Flask App ====================

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>接触角自动测量</title>
<style>
  :root {
    --bg: #f0f2f5;
    --card: #ffffff;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --text: #1e293b;
    --text-secondary: #64748b;
    --border: #e2e8f0;
    --success: #10b981;
    --danger: #ef4444;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
    --shadow-lg: 0 10px 25px rgba(0,0,0,.1);
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* Header */
  .header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    color: #fff;
    padding: 28px 24px;
    text-align: center;
  }
  .header h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: -.02em; }
  .header p  { margin-top: 6px; opacity: .85; font-size: .95rem; }

  /* Layout */
  .container {
    max-width: 960px;
    margin: 0 auto;
    padding: 32px 20px 60px;
  }

  /* Upload Zone */
  .upload-zone {
    background: var(--card);
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all .2s;
    position: relative;
    box-shadow: var(--shadow);
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--primary);
    background: #eff6ff;
  }
  .upload-zone.has-image {
    border-style: solid;
    border-color: var(--success);
    padding: 20px 24px;
  }
  .upload-icon {
    width: 64px; height: 64px;
    margin: 0 auto 16px;
    background: #eff6ff;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .upload-icon svg { width: 32px; height: 32px; color: var(--primary); }
  .upload-zone h3 { font-size: 1.1rem; margin-bottom: 4px; }
  .upload-zone .hint { color: var(--text-secondary); font-size: .85rem; }
  .upload-zone input[type=file] { display: none; }

  /* Preview */
  .preview-row {
    display: flex;
    gap: 16px;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
  }
  .preview-row img {
    max-height: 160px;
    border-radius: 8px;
    box-shadow: var(--shadow);
  }
  .preview-info {
    text-align: left;
    font-size: .85rem;
    color: var(--text-secondary);
    line-height: 1.6;
  }

  /* Buttons */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 28px;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    border: none;
    cursor: pointer;
    transition: all .15s;
  }
  .btn-primary {
    background: var(--primary);
    color: #fff;
  }
  .btn-primary:hover { background: var(--primary-hover); }
  .btn-primary:disabled {
    background: #94a3b8;
    cursor: not-allowed;
  }
  .btn-outline {
    background: #fff;
    color: var(--primary);
    border: 1.5px solid var(--primary);
  }
  .btn-outline:hover { background: #eff6ff; }

  .actions {
    margin-top: 24px;
    display: flex;
    gap: 12px;
    justify-content: center;
    flex-wrap: wrap;
  }

  /* Spinner */
  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin .6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Result Card */
  .result-card {
    background: var(--card);
    border-radius: var(--radius);
    box-shadow: var(--shadow-lg);
    margin-top: 32px;
    overflow: hidden;
  }
  .result-header {
    background: linear-gradient(135deg, #10b981, #059669);
    color: #fff;
    padding: 24px;
    text-align: center;
  }
  .result-header.error {
    background: linear-gradient(135deg, #ef4444, #dc2626);
  }
  .big-angle {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -.03em;
  }
  .big-angle .unit { font-size: 1.5rem; font-weight: 500; opacity: .8; }
  .angle-details {
    display: flex;
    gap: 32px;
    justify-content: center;
    margin-top: 12px;
    font-size: .95rem;
    opacity: .9;
  }
  .result-body { padding: 24px; }
  .result-body h4 {
    font-size: .95rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: .05em;
    margin: 20px 0 10px;
  }
  .result-body h4:first-child { margin-top: 0; }
  .result-body img {
    width: 100%;
    border-radius: 8px;
    box-shadow: var(--shadow);
    margin-bottom: 8px;
  }
  .meta-table {
    width: 100%;
    font-size: .85rem;
    border-collapse: collapse;
  }
  .meta-table td {
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
  }
  .meta-table td:first-child {
    color: var(--text-secondary);
    width: 140px;
  }

  /* Toast */
  .toast {
    position: fixed;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: #1e293b;
    color: #fff;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: .9rem;
    z-index: 999;
    animation: fadeIn .2s;
    pointer-events: none;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateX(-50%) translateY(-8px); } }

  /* Responsive */
  @media (max-width: 640px) {
    .header h1 { font-size: 1.3rem; }
    .big-angle { font-size: 2.8rem; }
    .angle-details { flex-direction: column; gap: 4px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>💧 接触角自动测量</h1>
  <p>上传液滴图片，全自动计算接触角 — 基于 Young-Laplace 方程拟合</p>
</div>

<div class="container">
  <!-- 上传区 -->
  <div class="upload-zone" id="uploadZone">
    <div id="uploadPrompt">
      <div class="upload-icon">
        <svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
        </svg>
      </div>
      <h3>点击上传或拖拽图片到此处</h3>
      <p class="hint">支持 JPG / PNG / BMP，建议液滴轮廓清晰的图片</p>
    </div>
    <div id="uploadPreview" style="display:none;"></div>
    <input type="file" id="fileInput" accept="image/*">
  </div>

  <!-- 按钮 -->
  <div class="actions">
    <button class="btn btn-primary" id="btnMeasure" disabled>
      开始测量
    </button>
    <button class="btn btn-outline" id="btnReset" style="display:none;">
      重新上传
    </button>
  </div>

  <!-- 结果区 -->
  <div id="resultArea"></div>
</div>

<script>
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const uploadPrompt = document.getElementById('uploadPrompt');
  const uploadPreview = document.getElementById('uploadPreview');
  const btnMeasure = document.getElementById('btnMeasure');
  const btnReset = document.getElementById('btnReset');
  const resultArea = document.getElementById('resultArea');
  let selectedFile = null;

  // 点击上传区
  uploadZone.addEventListener('click', () => fileInput.click());

  // 文件选择
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });

  // 拖拽
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(file) {
    if (!file.type.match(/image\/(jpeg|png|bmp|webp)/)) {
      showToast('请上传 JPG / PNG / BMP 格式的图片');
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (ev) => {
      uploadPrompt.style.display = 'none';
      uploadPreview.style.display = 'block';
      uploadPreview.innerHTML = `
        <div class="preview-row">
          <img src="${ev.target.result}" alt="preview">
          <div class="preview-info">
            <strong>${file.name}</strong><br>
            大小: ${(file.size / 1024).toFixed(1)} KB<br>
            类型: ${file.type}
          </div>
        </div>
      `;
      uploadZone.classList.add('has-image');
      btnMeasure.disabled = false;
      btnReset.style.display = 'inline-flex';
      resultArea.innerHTML = '';
    };
    reader.readAsDataURL(file);
  }

  // 重置
  btnReset.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    uploadPrompt.style.display = '';
    uploadPreview.style.display = 'none';
    uploadPreview.innerHTML = '';
    uploadZone.classList.remove('has-image');
    btnMeasure.disabled = true;
    btnReset.style.display = 'none';
    resultArea.innerHTML = '';
  });

  // 测量
  btnMeasure.addEventListener('click', async () => {
    if (!selectedFile) return;

    btnMeasure.disabled = true;
    btnMeasure.innerHTML = '<span class="spinner"></span> 处理中...';
    resultArea.innerHTML = '';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const resp = await fetch('/api/measure', { method: 'POST', body: formData });
      const data = await resp.json();

      if (!data.success) {
        resultArea.innerHTML = `
          <div class="result-card">
            <div class="result-header error">
              <div style="font-size:1.2rem;font-weight:700;">❌ 测量失败</div>
              <div style="margin-top:6px;opacity:.9;">${data.error}</div>
            </div>
          </div>`;
      } else {
        const r = data.result;
        resultArea.innerHTML = `
          <div class="result-card">
            <div class="result-header">
              <div class="big-angle">${r.angle_avg}<span class="unit">°</span></div>
              <div class="angle-details">
                <span>左侧: ${r.angle_left}°</span>
                <span>右侧: ${r.angle_right}°</span>
              </div>
            </div>
            <div class="result-body">
              <h4>📐 测量参数</h4>
              <table class="meta-table">
                <tr><td>液滴中心</td><td>(${r.cx}, ${r.cy})</td></tr>
                <tr><td>液滴半径</td><td>${r.radius} px</td></tr>
                <tr><td>基准线 Y</td><td>${r.baseline_y.toFixed(1)}</td></tr>
                <tr><td>轮廓点数</td><td>${r.contour_points}</td></tr>
              </table>

              <h4>🔍 ROI 增强对比</h4>
              <img src="data:image/png;base64,${r.roi_image}" alt="ROI对比">

              <h4>✂️ 分割结果</h4>
              <img src="data:image/png;base64,${r.seg_image}" alt="分割">

              <h4>📈 Young-Laplace 拟合</h4>
              <img src="data:image/png;base64,${r.yl_image}" alt="YL拟合">
            </div>
          </div>`;
      }
    } catch (err) {
      resultArea.innerHTML = `
        <div class="result-card">
          <div class="result-header error">
            <div style="font-size:1.2rem;font-weight:700;">❌ 网络错误</div>
            <div style="margin-top:6px;opacity:.9;">${err.message}</div>
          </div>
        </div>`;
    } finally {
      btnMeasure.disabled = false;
      btnMeasure.innerHTML = '开始测量';
    }
  });

  function showToast(msg) {
    const t = document.createElement('div');
    t.className = 'toast';
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2500);
  }
</script>
</body>
</html>'''


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/api/measure', methods=['POST'])
def api_measure():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '请上传图片'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'})

    # 保存上传文件
    import uuid
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        result = process_image(filepath)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        # 清理临时文件
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    print("=" * 50)
    print("  接触角自动测量服务已启动")
    print(f"  打开浏览器访问: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
