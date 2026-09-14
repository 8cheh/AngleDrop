# baseline-lemma · 基线识别（多算法模式 + 可视化 UI）

本目录是 **AngleDrop「Step to step」的第 3 步 —— baseline-lemma 基线识别**：
在给定液滴图像中，用**可切换的算法**识别台面/衬底基线

```
y = k * x + b
```

也就是三线接触点所在的固体表面直线。它是后续接触线检测、接触角拟合的地基：
基线错了，后面的角度全错。

> 第 1 步 Pre-process 的 `locate_drop` 里已经内置了一条基线（用于分割液滴），
> 但它只有一套固定流程。本步骤把「基线识别」拆出来，提供 **6 种可切换算法**，
> 每种都独立给出候选直线 + 置信度 + 证据，方便横向对比。

---

## 一、设计思路（独立实现，不复制 Opendrop 代码）

1. **基线是近水平直线**：限制斜率 `|k| ≤ max_slope`（默认 0.2，约 ±11.3°）。
2. **每种模式独立**：各自产生候选直线、置信度与「证据」（边缘点 / Hough 线段 / 行剖面曲线）。
3. **统一输出**：`(k, b)`、斜率角、液滴侧方向 `invert`、RMSE、内点数、置信度 Q。
4. **证据可视化**：除了最终叠加图，还输出算法内部看到的证据图和行剖面曲线，
   避免「黑箱」——可以清楚看到每个算法到底拟合了哪些点。

---

## 二、可选基线识别算法

| 模式 key | 含义 | 思路 | 特点 |
|---|---|---|---|
| `sobel_ransac` | Sobel 行梯度 + RANSAC | 行剖面峰值 → 逐列亚像素 → RANSAC | 经典稳健，默认模式 |
| `hough` | Hough 直线投票 | Canny + HoughLinesP，近水平线段打分 | 全局投票，抗局部断裂 |
| `otsu_column` | Otsu 列剖面界面 | 逐列 Otsu 找亮度两类分界，再拟合 | 阈值法，对亮度层次敏感 |
| `fitline` | 边缘点 Huber 拟合 | Canny 边缘点 + `cv2.fitLine(DIST_HUBER)` | 稳健最小二乘 |
| `edge_ransac` | 边缘点 RANSAC | Canny 边缘点 + 显式 RANSAC | 随机采样一致，抗离群 |
| `pca` | 边缘点 PCA 主方向 | 边缘点协方差主成分 = 基线方向 | 几何意义清晰，对离群较敏感 |

> `fitline` / `edge_ransac` / `pca` 三个「拟合类」算法共享同一套粗定位带内的
> Canny 边缘点，区别只在**估计器**不同；因此它们之间的对比，就是
> 「Huber 稳健最小二乘 vs 显式 RANSAC vs PCA」的估计器对比。

---

## 三、核心函数（`baseline_lemma.py`）

| 函数 | 作用 |
|---|---|
| `detect_baseline(bgr, loc=None, mode=..., max_slope, band, seed)` | **主函数**：按所选模式识别基线，返回参数/置信度/证据图/叠加图 |
| `annotate_baseline(...)` | 最终叠加图：内点 + 基线 + 液滴侧箭头 + 参数文字 |
| `process_baseline(bgr, ...)` | 预处理 + 基线识别完整流水线，供 UI 使用 |
| `_mode_sobel_ransac / _mode_hough / _mode_otsu_column / _mode_fitline / _mode_edge_ransac / _mode_pca` | 6 种模式实现 |
| `_ransac_line / _subpixel_peak_cols / _row_profile_abs / _line_polarity` | 内部几何/拟合工具 |

`detect_baseline` 成功时返回：

```
ok, mode, mode_label, k, b, invert, polarity, slope_deg, score,
rmse, quality, n_points, n_inliers, endpoints, reference,
evidence_vis, profile_vis, vis
```

其中 `reference` 是 Pre-process 内置基线与本次结果的偏差（角度差、中心偏移），
用来直观看到「专门做基线识别」比预处理内置基线强多少。

`loc` 参数可不传；为空时内部懒加载第 1 步的 `preprocess.locate_drop` 作为参考。

---

## 四、运行

依赖：`numpy opencv-python flask`（与第 1、2 步相同）

```bash
python app.py                 # 浏览器打开 http://127.0.0.1:5002
python app.py --port 8000
```

页面功能：

1. **上传图片**（点击/拖拽，可多选，自动选中第一张）；
2. **预处理**：可选超分辨率（2×/3×/4×），自动做液滴定位（作为参考基线）；
3. **基线算法**：下拉选择 6 种算法，拖动「最大斜率」「拟合搜索带」；
4. **运行**：展示 原图 → 超分图 → 液滴定位参考图 → 算法证据（行剖面 / 候选点）
   → 基线识别结果图，并输出直线方程、斜率角、液滴侧、RMSE、置信度、
   内点数，以及与预处理内置基线的偏差。

---

## 五、API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | UI 页面 |
| `/api/meta` | GET | `{sr_available, modes}` |
| `/api/upload` | POST | multipart 上传图片，返回 `[{id, name}]` |
| `/api/images` | GET | 已上传列表 |
| `/api/clear` | POST | 清空全部 |
| `/api/clear_upload` | POST | 删除单张 `{id}` |
| `/api/process` | POST | `{id, use_sr, sr_scale, mode, max_slope, band}` → 各阶段 base64 图 + 基线结果 |

---

## 六、与 Pre-process 内置基线的差异

- Pre-process 的 `locate_drop` 把「找基线」当作液滴分割的**中间步骤**，只保留
  一套固定流程（正极性 Sobel 行剖面 + RANSAC），且主要面向正立液滴；
  本目录把基线识别变成**独立步骤**，6 种算法可切换，正立/倒置都支持。
- 本目录输出**证据图与行剖面图**，可解释性更强；并额外给出 RMSE、
  置信度 Q、与预处理基线的偏差等量化指标。
- 所有代码为重新实现，未复制 Opendrop 源文件。
