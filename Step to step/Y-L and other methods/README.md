# Y-L and other methods · 接触角拟合

这是 AngleDrop「Step to step」的第 4 步，负责从液滴轮廓算出左右接触角。前面几步的输出在这里汇总：

```
第 1 步 Pre-process    → 液滴掩码 + 台面基线 (k, b) + 取向 invert
第 2 步 Sobel-lemma    → 左右接触点
第 3 步 baseline-lemma → 更精细的台面基线（可选）
第 4 步 本目录         → 用可切换的模型拟合轮廓，求左右接触角
```

结果导出成报表是第 5 步 `UI and output` 的事，这里只做核心拟合和可视化。

---

## 一、设计思路

1. 统一坐标系：把所有轮廓点变换到基线坐标系 `(r, z)`，`r` 沿基线向右，`z` 垂直基线指向液滴内部（`z>0` 是液相一侧）。正立、倒置液滴和倾斜基线共用一套公式。
2. 接触角符号约定：令接触点处切线向上为 `(t_r, t_z)`，则

   ```
   θ_left  = atan2(t_z,  t_r)
   θ_right = atan2(t_z, -t_r)
   ```

   全程不取绝对值，疏水液滴（θ>90°）也能正确表示。`abs(t_r)` 会把角度压到 ≤90°，是一个常见 bug。
3. 几何模型在接触点附近 `window_frac × 底宽` 的窗口内拟合（沿用 OpenDrop 的局部拟合思路），避免被重力压扁的整体轮廓污染接触角。
4. 物理模型 Young–Laplace 是整条液滴的形状方程，所以对整条单侧轮廓做全局拟合（ADSA），再读取接触线处的切线角。
5. 多种方法并列，每种独立给出接触角、残差 RMS 和拟合曲线，UI 里可以一键跑全部方法做横向对比。

---

## 二、可选拟合模型

| 模式 key | 含义 | 模型 | 特点 |
|---|---|---|---|
| `young_laplace` | Young–Laplace 双参数（ADSA） | ODE + 数值积分 | 含重力项 β |
| `young_laplace_cap` | Young–Laplace 单参数 | ODE，β=0 | 忽略重力，退化为球冠 |
| `circle` | 圆拟合 | Kasa 代数初值 + 几何距离 Gauss–Newton | 适合小液滴 |
| `ellipse` | 椭圆 | Taubin/AMS 拟合 | 更贴合非圆轮廓 |
| `poly2` | 二次多项式切线 | `r = f(z)` 最小二乘 | 局部切线 |
| `poly3` | 三次多项式切线 | `r = f(z)` 最小二乘 | 易过拟合 |
| `spline` | 三次平滑样条切线 | `UnivariateSpline` | 平滑、无需选阶数 |
| `line` | 直线（楔形） | 总体最小二乘 | 很平缓的接触线 |
| `auto` | 自适应 | 跑所有方法，按残差 + 复杂度惩罚择优 | 默认推荐用于批量 |

### Young–Laplace 方程

以顶点为原点、`z` 沿重力方向向下、弧长 `s` 为自变量，轴对称 Y-L 方程：

```
dX/dS = cos(φ)
dD/dS = sin(φ)
dφ/dS = 2b + λ·D − sin(φ)/X
```

其中 `b = L/R₀`（`L` 是长度尺度，`R₀` 顶点曲率半径），`λ = ρgL²/γ` 是无量纲毛细常数（Bond 数）。重力项 `+λD` 让液滴随深度增加逐渐摊平（正立液滴）。接触角 `θ = φ|_{D=1}`（`D=1` 对应接触线）。

实现要点：

- 顶点处 `sin(φ)/X` 是 0/0，用极限 `dφ/dS → b` 处理；
- 用 `scipy.integrate.solve_ivp`（RK45 + 事件检测）积分到 `D=1`；
- 同时监测 `φ ≥ ~185°`，防止参数病态时轮廓自卷成螺旋；
- 拟合用实测点到理论曲线的最近距离，配合多初值避免局部极小；
- 初值由圆拟合的接触角经球冠关系 `b ≈ 1 − cosθ` 给出。

---

## 三、核心函数（`contact_angle.py`）

| 函数 | 作用 |
|---|---|
| `detect_contact_angle(bgr, loc, contact, method=..., ...)` | 主函数：提取轮廓 → 分左右 → 按方法拟合 → 接触角 |
| `annotate_contact_angle(...)` | 结果叠加图：轮廓点 + 拟合曲线 + 切线 + 接触点 |
| `process_contact_angle(bgr, ...)` | 预处理 + 接触角拟合完整流水线，供 UI 使用 |
| `profile_points(gray, mask)` | 逐行极值 + 法向亚像素细化的轮廓提取 |
| `fit_circle / fit_ellipse / fit_poly / fit_line / fit_spline` | 各几何模型 |
| `young_laplace_profile(b, lam, ...)` | 数值积分 Y-L ODE |
| `contacts_from_mask(mask, k, b, invert)` | 未提供接触点时由掩码几何求交 |
| `angle_from_tangent(tr, tz, side)` | 切线向量 → 接触角（无绝对值） |

`detect_contact_angle` 成功时返回：

```
ok, method, method_label, requested_method, window_frac, v_frac,
left:  {ok, angle_deg, rms_px, n_points, r0, model},
right: {ok, angle_deg, rms_px, n_points, r0, model},
avg_angle, asymmetry, confidence,
baseline: {k, b, invert}, base_width, height, r_apex,
comparison: [{method, label, ok, left, right, avg, rms_px, n_left, n_right}, ...],
profile_rz, contacts, frame, vis
```

`loc` / `contact` 都能单独传入；`loc` 为空时内部懒加载第 1 步的 `locate_drop`，`contact` 为空时由掩码 + 基线几何求交。

---

## 四、运行

依赖：`numpy opencv-python flask scipy`（`scipy` 只被 Young–Laplace / 样条用到，缺失时这两类方法自动跳过）

```bash
python app.py                 # 浏览器打开 http://127.0.0.1:5003
python app.py --port 8000
python test_contact_angle.py  # 合成球冠 + YL 真值自检
```

页面功能：

1. 上传图片（点击/拖拽，可多选，自动选中第一张）；
2. 预处理：可选超分辨率（2×/3×/4×）；
3. 接触点：选择第 2 步 Sobel-lemma 的边缘算法与阈值方式；
4. 拟合方式：下拉选择 9 种模型，拖动局部窗口，勾选对比所有方法；
5. 运行：展示原图、超分图、液滴定位、接触点、拟合曲线，输出左右接触角、平均角、不对称度、置信度、底宽/液滴高、基线、Y-L 参数，以及一张全方法对比表。

---

## 五、API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | UI 页面 |
| `/api/meta` | GET | `{sr_available, methods, edge_modes, thresholds, have_sobel}` |
| `/api/upload` | POST | multipart 上传图片，返回 `[{id, name}]` |
| `/api/images` | GET | 已上传列表 |
| `/api/clear` | POST | 清空全部 |
| `/api/clear_upload` | POST | 删除单张 `{id}` |
| `/api/process` | POST | `{id, use_sr, sr_scale, method, window_frac, compare, edge_mode, threshold}` → 各阶段 base64 图 + 接触角结果 + 对比表 |

---

## 六、验证与已知限制

精度自检（`test_contact_angle.py`）：

- 解析球冠 θ=40/90/130°：`circle` 误差 < 0.8°；
- Young–Laplace 生成的轮廓（含重力项）：`young_laplace` 反演误差 < 1°，拟合出的 `b / λ` 与生成值一致。

与既有实现对照：把本目录的 `circle` / `poly2` / `line` 与仓库里 `cadrop`（CSIEC v5.3）的对应方法在 176 张实测图上对比，`circle` 与 `poly2` 的结果几乎逐张一致（差异 < 1°），说明轮廓提取与几何拟合的数学是对齐的。

已知限制：

1. 拟合结果依赖第 1、2 步的掩码与接触点质量；掩码在台面上被平切，底部约 3px 的轮廓缺失，对局部多项式/样条影响最大。
2. 扁平液滴（宽高比很大）上，`poly2/poly3/spline` 会因为平顶被带偏，角度系统性偏大；这类液滴建议用 `circle` 或 `young_laplace`。
3. `young_laplace` 对轮廓噪声与初值敏感；本实现用多初值 + 事件检测缓解，但仍可能出现左右不对称。
4. `ellipse` 是 5 参数模型，局部窗口较小时容易过拟合——UI 的对比表就是为了让这种差异暴露出来。
5. 所有代码为重新实现，未复制 Opendrop / cadrop 源文件。
