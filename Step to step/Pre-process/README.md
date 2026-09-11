# Pre-process · 超分辨率 + 液滴定位（可视化 UI）

本目录是 **AngleDrop「Step to step」的第 1 步 —— 预处理**：只做
**超分辨率（可选）+ 液滴定位**，不做后续的基线细化、拟合、求角度等步骤。

---

## 一、核心函数（`preprocess.py`）

| 函数 | 来源 | 作用 |
|---|---|---|
| `read_image(path)` | `_src/cadrop_detect.py`（一致） | 读取 BGR，容忍 Windows 非 ASCII 路径 |
| `sr_available()` | **GitHub 独有** `cadrop/superres.py` | 检测 FSRCNN 超分是否可用 |
| `super_resolve(bgr, scale)` | **GitHub 独有** `cadrop/superres.py` | 液滴 ROI 超分 + 其余快速插值 |
| `locate_drop(bgr)` | 移植 `cadrop_detect.py` 的 `find_substrate` + `_segment` | 联合搜索「台面基线 + 液滴」，输出包围盒/掩码/基线 |
| `process(bgr, ...)` | 本目录新增 | 把「超分 + 定位」串成流水线 |

### 与 GitHub 仓库的差异

本地 `_src/` 只有 8 个文件，**没有** `superres.py`。GitHub 仓库
`All in one/Old versions/Summer vacation ver 5.3 (available)/cadrop/` 多出了：

- `cadrop/superres.py` —— FSRCNN 超分辨率（只在液滴 ROI 跑网络，其余快速插值）；
- `cadrop/measure.py` 里的 `use_sr` / `sr_scale` —— 把超分作为测量管线第一步预处理。

本目录把 `superres.py` 的算法搬过来并做了适配：

1. **无 `dnn_superres` 时回退**：当前环境是 `opencv-python`（无 contrib），自动回退到
   「三次插值 + `detailEnhance` + 反锐化」；装了 `opencv-contrib-python` 并把
   `FSRCNN_x3.pb` 等模型放进 `models/` 后自动走神经网络路径。
2. **`locate_drop`** 把 `cadrop_detect.py` 里散在 `find_substrate` / `_segment` 的
   定位逻辑整理成一个独立函数，返回包围盒、掩码、基线参数。

---

## 二、流程

```
上传图片 → 超分辨率(可选, FSRCNN/经典回退) → 液滴定位(基线 + 包围盒 + 掩码)
```

---

## 三、运行

依赖：`numpy opencv-python flask`

```bash
python app.py                 # 浏览器打开 http://127.0.0.1:5000
python app.py --port 8000
```

页面功能：

1. **上传图片**（点击/拖拽，可多选，自动选中第一张）；
2. **开关超分**并选放大倍数（2×/3×/4×）；
3. **运行**：展示 原图 → 超分图 → 定位结果图（包围盒 + 基线 + 掩码高亮），
   并输出包围盒坐标、尺寸、面积、基线方程。

---

## 四、API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | UI 页面 |
| `/api/meta` | GET | `{sr_available: bool}` |
| `/api/upload` | POST | multipart 上传图片，返回 `[{id, name}]` |
| `/api/images` | GET | 已上传列表 |
| `/api/clear` | POST | 清空全部 |
| `/api/clear_upload` | POST | 删除单张 `{id}` |
| `/api/process` | POST | `{id, use_sr, sr_scale}` → 返回各阶段 base64 图 + 定位结果 |
