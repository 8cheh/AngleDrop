**AngleDrop**
--------------------------

联系: huangbache@outlook.com

# 简介
AngleDrop / AngleAP 是一套用于从单张图像自动测量 sessile 液滴接触角的工具集合，自动处理液滴检测、基线（基底）定位、轮廓提取与角度拟合。该仓库同时包含“Step to step”（分步可视化/教学型）与“All in one”（一体化/历史版本）两种代码组织方式，便于开发调试与对比算法实现。

# 当前工作（Overview of current work）
- 主要完成：预处理模块（Step to step/Pre-process）已实现一个浏览器可视化界面，用于上传图片并做“超分辨率（可选） + 液滴定位（基线 + 包围盒 + 掩码）”的流水线演示。
- 测量管线：在 All in one 的旧版本代码中保留了端到端测量流程（见 `All in one/Old versions/.../cadrop/measure.py`），包括增强、基线搜索、轮廓/剖面提取以及左右侧拟合与角度计算，作为参考实现与算法基线。
- 超分辨率支持：工程内提供可选的 ROI 超分模块（FSRCNN），运行时会在检测到 `opencv-contrib-python` 与模型文件时启用。若环境不具备 DNN 支持，回退为经典插值 + `detailEnhance` / 反锐化的处理路径。

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

# 重要实现细节
- 坐标系与约定：内部使用基线坐标 (r, z)，z=0 为基线，r 指径向坐标。角度计算通过将切线向上取向并区分左右侧符号实现，支持 >90° 的疏水情况。
- Circle 拟合：初始代数解（Kasa）+ 几何 Gauss-Newton 迭代精化，返回 (rc, zc, R, rms)。
- Ellipse / Conic：使用 Halir–Flusser 方式直接拟合二次曲线系数（A..F），并在 z=0 解交点以求切线。
- 多项式拟合：以 r = f(z) 的多项式拟合并在 z=0 计算斜率。
- 置信度计算：综合左右拟合 RMS 相对于基宽、单侧点数、左右一致性（非对称性）构成一个 [0,1] 的置信度值。

# 代码位置（快速导航）
- Step to step/Pre-process/: 预处理与可视化 UI（上传、超分开关、定位结果展示）。
  - preprocess.py（核心流水线，read_image, super_resolve, locate_drop, process）
  - app.py（Flask UI）
- All in one/Old versions/.../cadrop/: 一体化的历史实现（端到端测量）：
  - measure.py（端到端测量流程、SR 集成、结果封装）
  - fitting.py（拟合与角度提取的数学实现）
  - detect.py（基线与掩码、轮廓、剖面提取；在 measure.py 被导入使用）
  - superres.py（FSRCNN 超分实现，若存在则被 measure.py / preprocess 调用）

# 如何运行（Step to step 预处理 UI）
依赖：numpy, opencv-python, flask

```bash
python "Step to step/Pre-process/app.py"    # 启动浏览器 UI，默认 http://127.0.0.1:5000
python "Step to step/Pre-process/app.py" --port 8000
```
页面功能：上传图片；可选启用超分（2×/3×/4×）；运行并查看原图→超分图→定位结果（包围盒、基线、掩码）与定位数值输出。

# 版本说明与建议
- 仓库保留了“教学/分步”与“历史一体化”两个分支形式：若你想调试或逐步验证某一环节（例如只看超分或只看定位），建议使用 Step to step 下的预处理 UI；若要跑完整批处理测量（包含拟合/置信度输出），参考 All in one/Old versions/.../cadrop/ 下的 measure.py。
- 若要启用神经网络超分，请安装 `opencv-contrib-python` 并把模型文件（例如 FSRCNN_x3.pb 等）放入 `models/` 目录。没有 contrib 时会自动采用经典回退策略。

---
更新历史: 本次 README 扩展，补充了当前正在维护的“预处理”工作说明，并将 All-in-one 旧版测量算法的逻辑流程与实现细节（拟合方法、角度约定、SR 行为）进行了总结，便于新用户快速定位代码与理解算法链路。
