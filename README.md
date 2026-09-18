# Super-resolution (SR) 安装与模型文件 / Super-resolution (SR) install & model files

下面说明如何在本地启用基于 OpenCV 的 DNN 超分（FSRCNN）。如果不启用或环境不支持 DNN，仓库的代码会自动回退到经典插值 + 细节增强（detailEnhance）/反锐化流程。

1) 安装 OpenCV Contrib（含 dnn_superres）：

```bash
# 如果你在桌面环境或需要 GUI：
pip install opencv-contrib-python

# 在无 GUI 的服务器上可选：
pip install opencv-contrib-python-headless
```

注意：替换系统 Python/虚拟环境或 Conda 环境中的 opencv-python 时请确保先卸载旧版以避免冲突。

2) 模型文件（FSRCNN）

- 建议的模型文件名（放在仓库根目录的 models/ 目录中）：
  - models/FSRCNN_x2.pb
  - models/FSRCNN_x3.pb
  - models/FSRCNN_x4.pb

这些预训练模型通常可以从 OpenCV 的 DNN Super-Resolution 模型集合或作者提供的模型仓库下载（搜索 "FSRCNN pb model" 可找到来源）。将需要的 .pb 文件放入项目的 models/ 目录后，代码会自动发现并启用相应倍数的 SR。

3) 快速测试（Python snippet）

以下片段演示如何用 OpenCV 的 dnn_superres 加载并运行 FSRCNN（在安装了 opencv-contrib 后）：

```bash
python - <<'PY'
import cv2
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('models/FSRCNN_x3.pb')
sr.setModel('fsrcnn', 3)
img = cv2.imread('input.jpg')
res = sr.upsample(img)
cv2.imwrite('out_sr.jpg', res)
print('wrote out_sr.jpg')
PY
```

4) 在本项目中如何启用

- Step to step/Pre-process UI：页面有“Enable SR”开关与 2×/3×/4× 选项，启用后会尝试读取 models/ 下对应文件并运行 ROI 超分。
- All in one / measure.py：在调用 measure(..., use_sr=True, sr_scale=3.0) 时会先在原图上定位 ROI，再对 ROI 做放大（使用 superres.upscale_roi 或回退插值）。

5) 常见问题与故障排查

- 报错 "module 'cv2' has no attribute 'dnn_superres'"：说明当前安装的 opencv 版本不包含 contrib 模块，执行 `pip install opencv-contrib-python`。
- 模型找不到：确认 models/ 目录存在且模型文件名与代码/UI 中的倍数匹配（FSRCNN_x3.pb 对应 scale=3）。
- 性能/显存：SR 在较大 ROI 或较高倍数时会占用较多内存，UI 中提供了经典回退以保证能在没有 DNN 硬件下也能处理图像。

---

Super-resolution (SR) install & model files (English summary)

This project supports OpenCV DNN-based FSRCNN super-resolution for droplet ROI upscaling. If DNN support or model files are absent, the code falls back to bicubic interpolation + detail enhancement and unsharp operations.

- Install OpenCV contrib (contains dnn_superres):
  - pip install opencv-contrib-python  (or opencv-contrib-python-headless for servers)

- Place model files under the repository `models/` directory, for example:
  - models/FSRCNN_x2.pb
  - models/FSRCNN_x3.pb
  - models/FSRCNN_x4.pb

- Quick test (run after installing opencv-contrib):

```bash
python - <<'PY'
import cv2
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('models/FSRCNN_x3.pb')
sr.setModel('fsrcnn', 3)
img = cv2.imread('input.jpg')
res = sr.upsample(img)
cv2.imwrite('out_sr.jpg', res)
print('wrote out_sr.jpg')
PY
```

- How to enable in this project:
  - Step to step/Pre-process UI: toggle "Enable SR" and choose 2×/3×/4×; the UI will look for matching models in models/.
  - All in one / measure.py: call measure(..., use_sr=True, sr_scale=<2|3|4>) to enable SR in the pipeline.

- Troubleshooting tips:
  - If cv2.dnn_superres is missing, install opencv-contrib-python.
  - Ensure model filenames match requested scale.
  - SR can be memory-intensive; use the fallback mode for large images or constrained environments.

