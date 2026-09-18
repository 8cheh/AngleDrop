# UI and output · Step to step 完整流水线（总结）

本目录是 **AngleDrop「Step to step」的收口**：把前面四步串成**一条可配置的
流水线**，并提供一个完整的 Web UI。**每一步都可以自选方法**，最终输出
接触角结果、各阶段图像、以及可导出的 JSON / CSV / 标注图。

```
                 ┌────────────────────────────────────────────────────────────┐
   输入图像 ──►  ① Pre-process        超分辨率（可选）+ 液滴定位
                 │                                                            │
                 ▼                                                            │
             ② baseline-lemma       台面基线识别（6 种算法，可选）
                 │                                                            │
                 ▼                                                            │
             ③ Sobel-lemma          接触线 / 接触点（6 种边缘 × 3 种阈值）
                 │                                                            │
                 ▼                                                            │
             ④ Y-L and other        接触角拟合（9 种模型，可选）
                 │                                                            │
                 ▼                                                            │
             输出：角度 + 分阶段图 + 对比表 + JSON/CSV/PNG 导出  ◄─────────────┘
```

---

## 一、目录里有什么

| 文件 | 说明 |
|---|---|
| `pipeline.py` | **流水线编排核心**：把四步串起来，每步可选方法；结果整理/导出 |
| `app.py` | Flask 页面与 API（默认端口 5004） |
| `templates/index.html` | 完整 UI：分步配置 + 结果总览 + 阶段图筛选 + 导出 |
| `README.md` | 本文件 |

`pipeline.py` 通过把四个子目录加入 `sys.path`，
直接复用：

```python
import preprocess as pp        # 第 1 步
import baseline_lemma as bl    # 第 2 步
import sobel_lemma as sl       # 第 3 步
import contact_angle as ca     # 第 4 步
```

---

## 二、每一步可选的方法

| 步骤 | 可选方法 | 来源 |
|---|---|---|
| ① 预处理 | 超分开关 + 2×/3×/4×（FSRCNN / 经典回退） | `preprocess.py` |
| ② 基线识别 | `preprocess`（内置基线）、Sobel+RANSAC、Hough、Otsu 列剖面、Huber 拟合、边缘 RANSAC、PCA | `baseline_lemma.py` |
| ③ 接触线 | Sobel / Scharr / Laplacian / Sobel+Laplacian / LoG / Canny ×（Otsu / P88 / P95） | `sobel_lemma.py` |
| ④ 接触角 | Young–Laplace（双/单参数）、圆、椭圆、二次/三次多项式、样条、直线、自适应 | `contact_angle.py` |

**方法选择如何影响下游**：第 ② 步若选择除 `preprocess` 以外的方法，
会用新基线**重新分割液滴**（`_rebuild_loc`），保证第 ③、④ 步使用的掩码
与新基线一致——避免「基线换了、掩码没换」造成的错位。

---

## 三、核心函数（`pipeline.py`）

| 函数 | 作用 |
|---|---|
| `run_pipeline(bgr, **config)` | 跑完整四步，返回分步结果 + 阶段图 |
| `result_to_dict(result, name)` | 整理为可 JSON 序列化的精简结果 |
| `result_to_csv(result, name)` | 整理为单行 CSV（含表头） |
| `METHOD_CATALOG` | 各步可选方法总目录，UI 下拉框直接使用 |

`run_pipeline` 成功时返回：

```python
{
  'ok': bool,
  'config': {...},                       # 本次使用的方法与参数
  'steps': {
      'preprocess': {ok, sr_active, bbox, area, baseline},
      'baseline':   {ok, mode, mode_label, k, b, invert, rmse, quality,
                     score, n_inliers, reference},
      'contact':    {ok, mode, mode_label, threshold,
                     left, right, span, arc_length},
      'angle':      {ok, method, method_label, left, right, avg_angle,
                     asymmetry, confidence, base_width, height, yl,
                     comparison:[...]},
  },
  'stages': [(step, name, bgr, desc), ...],   # 各阶段图
  'image_size': [w, h],
  'error': str | None,
}
```

---

## 四、运行

依赖：`numpy opencv-python flask scipy`（与前面几步一致）

```bash
python app.py                 # 浏览器打开 http://127.0.0.1:5004
python app.py --port 8000
```

页面功能：

1. **上传图片**（点击/拖拽，可多选，列表中标 ✅ 表示已有结果）；
2. **① 预处理**：超分开关 + 倍数；
3. **② 基线识别**：下拉选算法（含「预处理内置基线」），调最大斜率 / 搜索带；
4. **③ 接触线**：选边缘算法 + 阈值方式 + 接触点搜索带；
5. **④ 接触角**：选拟合模型 + 局部窗口，可勾选「对比所有拟合方法」；
6. **运行**：右侧「结果总览」给出四步的状态卡与关键数值、平均接触角、
   各模型对比表；下方按步骤筛选查看所有阶段图；
7. **导出**：结果 JSON / CSV、标注图 PNG；「批量跑全部图片」用当前方法
   处理所有已上传图片并下载汇总 CSV。

---

## 五、API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | UI 页面 |
| `/api/meta` | GET | `{sr_available, catalog:{baseline,edge,threshold,angle}}` |
| `/api/upload` | POST | multipart 上传，返回 `[{id, name}]` |
| `/api/images` | GET | 已上传列表（含 `has_result`） |
| `/api/clear` | POST | 清空全部 |
| `/api/clear_upload` | POST | 删除单张 `{id}` |
| `/api/process` | POST | 运行完整流水线，返回分步结果 + base64 阶段图 |
| `/api/export?id=&fmt=json\|csv` | GET | 下载结果 JSON / CSV |
| `/api/export_image?id=` | GET | 下载标注图 PNG |
| `/api/batch` | POST | 用当前配置批量处理全部图片，返回汇总 CSV + 摘要 |

`/api/process` 请求体示例：

```json
{
  "id": "abc123",
  "use_sr": false, "sr_scale": 3,
  "baseline_mode": "sobel_ransac", "max_slope": 0.2, "band": 60,
  "edge_mode": "canny", "threshold": "otsu", "contact_band": 10,
  "angle_method": "young_laplace", "window_frac": 0.25, "compare": true
}
```

---

## 六、复现结果与说明

- 在 176 张实测液滴图上，默认配置
  （`baseline_mode=sobel_ransac` + `edge=sobel` + `angle=auto`）
  的成功率为 **40/40**（抽样 40 张全部成功）；
- 第 4 步的 `circle` / `poly2` 与仓库既有实现 `cadrop`（CSIEC v5.3）
  在同样图片上几乎逐张一致（差异 < 1°），说明轮廓提取与几何拟合对齐；
- `auto` 采用「模型偏好顺序 + 相对残差 + 单侧失败惩罚 + 不对称惩罚」，
  优先选择稳定的 `circle` / `young_laplace`，避免样条/高次多项式过拟合
  （否则会在扁平液滴上给出 150°+ 的假角度）。

**已知限制**：整条流水线的精度受上游制约——第 ① 步的掩码在台面上被平切，
底部约 3px 轮廓缺失；第 ② 步部分算法（如 Hough）在个别图上会失败并自动
回退；第 ④ 步的 `ellipse/样条` 在窗口较小时易过拟合。这些差异正是本 UI
「方法对比」想暴露的：**换一个方法，答案会变**，而这本身就是教学重点。

所有代码为重新实现，未复制 Opendrop 源文件。
