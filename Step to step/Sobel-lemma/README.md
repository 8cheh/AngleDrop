# Sobel-lemma · 接触线检测（多边缘算法模式）

本目录是 **AngleDrop「Step to step」的第 2 步 —— Sobel-lemma 接触线检测**。
它建立在第 1 步 **Pre-process** 的输出之上：预处理已经给出液滴掩码、包围盒、
台面基线 `(k, b)`；本步骤用 **可切换的边缘检测算法** 重新识别液滴轮廓，
并在轮廓与基线的交点处给出 **左右接触点**（三线接触点在 2D 图中的投影）。

> 只做接触线检测核心 + 独立可视化 UI，不负责与后续「角度拟合」步骤的接口拼接。

---

## 一、设计思路（独立实现，不复制 Opendrop 代码）

Opendrop 的整体思路是「先找基线 → 再找液滴 → 最后在基线与液滴轮廓的交点求接触点」。
本目录借鉴的是这个**流程思想**，但实现全部自研：

1. **每种模式先生成边缘响应图**：`edge_strength` 给出 float32 强度，
   `edge_binary` 用 Otsu / 分位阈值转成 0/255 二值边缘。
2. **用预处理掩码约束边缘**：`mask_band` 提取液滴掩码的边界带，
   把边缘检测限定在真实的液-气界面上，排除台面纹理、倒影与内部噪声。
3. **接触点 = 轮廓上离基线最近且边缘最强的点**：先由掩码估计粗略接触点，
   再在所选算法的边缘图上做局部搜索 + 亚像素加权质心细化，最后把 y 落到基线上
   （三线接触点位于固体表面）。
4. **轮廓弧**：左右接触点之间的液滴轮廓，优先取边缘图的连通路径；
   边缘断裂时回退到「掩码轮廓吸附到最近边缘像素」的混合路径，再退到掩码轮廓。

---

## 二、可选边缘算法模式

| 模式 key | 含义 | 特点 |
|---|---|---|
| `sobel` | Sobel 梯度幅值 | 对强边缘稳定，默认模式 |
| `scharr` | Scharr 梯度幅值 | 对斜向边缘更敏感 |
| `laplacian` | Laplacian 二阶导幅值 | 对灰度突变响应强，噪声也较敏感 |
| `sobel_laplacian` | Sobel + Laplacian 归一化融合 | 一阶/二阶信息互补 |
| `log` | LoG（Marr-Hildreth 零交叉） | 高斯平滑后求零交叉，边缘较连续 |
| `canny` | Canny 自适应双阈值 | 抑制弱边缘，边缘细且连续 |

二值化阈值方式（`THRESHOLD_MODES`）：

| key | 含义 |
|---|---|
| `otsu` | Otsu 自适应阈值（默认） |
| `p88` | 88% 分位阈值 |
| `p95` | 95% 分位阈值 |

---

## 三、核心函数（`sobel_lemma.py`）

| 函数 | 作用 |
|---|---|
| `edge_strength(gray, mode)` | 返回 float32 边缘强度图 |
| `edge_binary(gray, mode, threshold, mask)` | 返回 0/255 二值边缘图 |
| `mask_band(mask, inner_r, outer_r)` | 液滴掩码的边界带 |
| `approximate_contacts(mask, k, b, invert)` | 由掩码估计粗略接触点 |
| `detect_contact_line(bgr, loc, mode, ...)` | **主函数**：检测接触线，返回接触点/跨度/弧长/边缘图/可视化 |
| `annotate_contact_line(...)` | 画边缘轮廓、接触点、基线、跨度 |
| `process_contact_line(bgr, ...)` | 预处理 + 接触线检测完整流水线，供 UI 使用 |

`detect_contact_line` 返回字段（成功时）：

```
ok, mode, mode_label, threshold, threshold_label,
edge, strength, outline,
left:  {x, y, z, conf, edge:{x,y}},
right: {x, y, z, conf, edge:{x,y}},
span, arc_length,
baseline: {k, b, invert},
vis
```

`loc` 参数可不传；为空时内部懒加载第 1 步的 `preprocess.locate_drop`。

---

## 四、运行

依赖：`numpy opencv-python flask`（与第 1 步相同）

```bash
python app.py                 # 浏览器打开 http://127.0.0.1:5001
python app.py --port 8000
```

页面功能：

1. **上传图片**（点击/拖拽，可多选，自动选中第一张）；
2. **预处理**：可选超分辨率（2×/3×/4×），自动做液滴定位；
3. **接触线检测模式**：下拉选择 6 种边缘算法 + 3 种阈值方式，拖动接触点搜索带；
4. **运行**：展示 原图 → 超分图 → 液滴定位图 → 所选算法边缘图 → 接触线检测结果图，
   并输出左右接触点坐标、离基线距离、接触跨度、轮廓弧长、基线方程。

---

## 五、API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | UI 页面 |
| `/api/meta` | GET | `{sr_available, modes, thresholds}` |
| `/api/upload` | POST | multipart 上传图片，返回 `[{id, name}]` |
| `/api/images` | GET | 已上传列表 |
| `/api/clear` | POST | 清空全部 |
| `/api/clear_upload` | POST | 删除单张 `{id}` |
| `/api/process` | POST | `{id, use_sr, sr_scale, mode, threshold, contact_band}` → 各阶段 base64 图 + 接触线结果 |

---

## 六、与 Opendrop 的差异

- Opendrop 使用固定的一套图像处理流程并直接给出接触角结果；
  本目录把「接触线检测」拆成独立步骤，并让**边缘算法可切换**，便于对比不同算子的表现。
- Opendrop 的接触点主要靠轮廓/灰度分析；本目录用「掩码拓扑 + 边缘强度局部搜索 +
  亚像素细化」的混合策略，既抗断裂，又能体现不同边缘算子的差异。
- 所有代码为重新实现，未复制 Opendrop 源文件。
