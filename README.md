**AngleDrop**
--------------------------

联系 / Contact: huangbache@outlook.com

# 简介
AngleDrop / AngleAP 是一套用于从单张图像自动测量 sessile 液滴接触角的工具集合，自动处理液滴检测、基线（基底）定位、轮廓提取与角度拟合。该仓库同时包含“Step to step”（分步可视化/教学型）与“All in one”（一体化/历史版本）两种代码组织方式，便于开发调试与对比算法实现。

# 简介（English）
AngleDrop / AngleAP is a toolkit for automatically measuring the contact angle of a sessile drop from a single image. It handles droplet detection, substrate baseline localization, contour extraction, and angle fitting. The repository contains both a "Step to step" interactive/educational preprocessing UI and an "All in one" historical end-to-end implementation for development and comparison.

# 当前工作（Overview of current work）
- 主要完成：预处理模块（Step to step/Pre-process）已实现一个浏览器可视化界面，用于上传图片并做“超分辨率（可选） + 液滴定位（基线 + 包围盒 + 掩码）”的流水线演示。
- 测量管线：在 All in one 的旧版本代码中保留了端到端测量流程（见 `All in one/Old versions/.../cadrop/measure.py`），包括增强、基线搜索、轮廓/剖面提取以及左右侧拟合与角度计算，作为参考实现与算法基线。
- 超分辨率支持：工程内提供可选的 ROI 超分模块（FSRCNN），运行时会在检测到 `opencv-contrib-python` 与模型文件时启用。若环境不具备 DNN 支持，回退为经典插值 + `detailEnhance` / 反锐化的处理路径。

# Current work (English)
- Preprocess UI: The Step to step/Pre-process module provides a browser UI for uploading images and running an optional "super-resolution (SR) + droplet localization (baseline + bounding box + mask)" pipeline.
- Measurement pipeline: The All in one historical code (cadrop/measure.py) contains an end-to-end measurement pipeline (enhancement, substrate search, contour/profile extraction, left/right-side fits and angle calculation) which we keep as a reference implementation.
- Super-resolution support: An optional ROI-SR module (FSRCNN) is available and will be used when `opencv-contrib-python` and the model files are present; otherwise the code falls back to classic interpolation + detailEnhance / unsharp operations.

# 算法逻辑流程（总体）
1. 读取图像（支持 Windows 非 ASCII 路径）。
2. （可选）超分辨率：在检测到液滴 ROI 后对 ROI 做放大重建；其余区域做快速插值，避免整体超分带来的计算开销。
3. 图像增强（可选）：用于改善边缘/对比，提升轮廓检测鲁棒性。
4. 基线（substrate）检测：搜索台面线并建立基线参数（k, b），将图像坐标变换到基线坐标系（r, z，z=0 为基线）。
5. 液滴分割/掩码：提取液滴区域并计算外轮廓以得到剖面点集合。
6. 剖面点抽取与筛选：在基线坐标系中取位于基线以上的点，按左右两侧分割，选取靠近接触线的窗口用于拟合。
7. 单侧拟合（fit one side）：对每一侧在窗口内尝试不同模型并选择最优：
   - circle（圆弧拟合 + Gauss-Newton 精化）
   - ellipse / conic（Halir–Flusser 直接椭圆拟合，计算与基线交点与切线）
   - poly（二次多项式 r(z)）
   - line（线性拟合，作为小角近似/直边）
   - auto：优先判定为直线（若残差很小），否则尝试圆拟合
8. 接触角计算：通过拟合曲线在 z=0 处的切线方向，使用统一的切线到角度转换约定（见 fitting.py）计算角度（theta_left / theta_right，取范围 (0, 180)°）。
9. 结果组合：生成左右侧角度、平均角、非对称性、基线参数、接触点坐标、拟合曲线用于可视化，并计算置信度评分（基于 RMS、点数与左右一致性）。

# Algorithm flow (English)
1. Read image (supports Windows paths with non-ASCII characters).
2. (Optional) Super-resolution: after locating the droplet ROI, upscale the ROI; the rest of the image is resampled quickly to avoid expensive full-image SR.
3. (Optional) Image enhancement to improve edge/contrast for better contour detection.
4. Substrate baseline detection: find the table line and compute baseline parameters (k, b), transform image coordinates into baseline coordinates (r, z) with z=0 at the baseline.
5. Droplet mask / contour extraction: get the droplet region and external contour to produce profile points.
6. Profile extraction & selection: keep points above the baseline, split into left/right halves, and select a window near the contact line for fitting.
7. Fit one side: try multiple models on each side and pick the best:
   - circle (Kasa algebraic start + geometric Gauss–Newton refinement)
   - ellipse / conic (Halir–Flusser direct conic fit, solve intersection at z=0 and compute tangent)
   - poly (quadratic polynomial r(z))
   - line (linear fit, small-angle approximation)
   - auto: prefer line if residuals are tiny, else try circle
8. Contact-angle extraction: compute angle from the tangent at z=0 using the shared tangent→angle convention in fitting.py; angles are in (0, 180)°.
9. Result aggregation: left/right angles, mean angle, asymmetry, baseline, contact coordinates, fitted curves for visualization, and a confidence score based on RMS, point counts and L/R agreement.

# 重要实现细节 / Implementation details
- 坐标系与约定：内部使用基线坐标 (r, z)，z=0 为基线，r 指径向坐标。角度计算通过将切线向上取向并区分左右侧符号实现，支持 >90° 的疏水情况。
- Circle 拟合：初始代数解（Kasa）+ 几何 Gauss-Newton 迭代精化，返回 (rc, zc, R, rms)。
- Ellipse / Conic：使用 Halir–Flusser 方式直接拟合二次曲线系数（A..F），并在 z=0 解交点以求切线。
- 多项式拟合：以 r = f(z) 的多项式拟合并在 z=0 计算斜率。
- 置信度计算：综合左右拟合 RMS 相对于基宽、单侧点数、左右一致性（非对称性）构成一个 [0,1] 的置信度值。

# Important implementation details (English)
- Coordinates & conventions: Internally the code uses baseline coordinates (r, z) with z=0 at the substrate; r is radial. Angles are computed by orienting the tangent upwards and distinguishing left/right signs, allowing hydrophobic angles (>90°).
- Circle fit: algebraic Kasa start followed by geometric Gauss–Newton refinement; returns (rc, zc, R, rms).
- Ellipse / conic: Halir–Flusser direct conic fit (A..F), solve for intersection with z=0 and compute tangent.
- Polynomial fit: r = f(z) polynomial, slope at z=0 gives tangent.
- Confidence score: combined from RMS relative to base width, number of points, and left/right agreement producing a value in [0,1].

# 代码位置（快速导航） / Code locations
- Step to step/Pre-process/: 预处理与可视化 UI（上传、超分开关、定位结果展示）。
  - preprocess.py（核心流水线，read_image, super_resolve, locate_drop, process）
  - app.py（Flask UI）
- All in one/Old versions/.../cadrop/: 一体化的历史实现（端到端测量）：
  - measure.py（端到端测量流程、SR 集成、结果封装）
  - fitting.py（拟合与角度提取的数学实现）
  - detect.py（基线与掩码、轮廓、剖面提取；在 measure.py 被导入使用）
  - superres.py（FSRCNN 超分实现，若存在则被 measure.py / preprocess 调用）

# How to run (shortest path)
- 预处理 UI（Step to step/Pre-process）依赖：numpy, opencv-python, flask

```bash
python "Step to step/Pre-process/app.py"    # 启动浏览器 UI，默认 http://127.0.0.1:5000
python "Step to step/Pre-process/app.py" --port 8000
```
页面功能：上传图片；可选启用超分（2×/3×/4×）；运行并查看原图→超分图→定位结果（包围盒、基线、掩码）与定位数值输出。

# 超分（SR）注意事项 / SR notes
- 若要启用神经网络超分，请安装 `opencv-contrib-python` 并把 FSRCNN 等模型文件放入 `models/` 目录（或代码中配置的模型路径）。
- 在没有 DNN 支持时，代码会自动回退到经典插值 + detailEnhance/反锐化 的方法。

# 建议的后续工作 / Suggestions
- 补充一张“算法流程图”或序列图到 docs/ 以便读者快速理解数据流。
- 增加示例图像和可复现的测试套件（单元/集成），便于回归验证不同 SR/拟合策略的影响。
- 将 measure.py 中的多个魔数/阈值提为配置参数并暴露 CLI，以便批处理测量。

---
更新历史: 本次 README 扩展，补充了当前正在维护的“预处理”工作说明，并将 All-in-one 旧版测量算法的逻辑流程与实现细节（拟合方法、角度约定、SR 行为）进行了总结，便于新用户快速定位代码与理解算法链路。
