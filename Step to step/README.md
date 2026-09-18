# Step to step · 接触角测量（分步复现）

把「测量一张液滴图像的接触角」拆成**五个可独立运行的小步骤**，
每一步都有自己的算法、可视化 UI 与 README，最后用
`UI and output` 把它们串成一条完整流水线。

> 面向教学：每一步都可以**换算法**，用同一张图对比不同方法的差异。

```
输入图像
   │
   ▼
① Pre-process          超分辨率（可选）+ 液滴定位（基线 + 掩码 + 包围盒）
   │                    python app.py   →  http://127.0.0.1:5000
   ▼
② Sobel-lemma          接触线 / 左右接触点（6 种边缘算法 × 3 种阈值）
   │                    python app.py   →  http://127.0.0.1:5001
   ▼
③ baseline-lemma       台面基线识别（6 种算法：梯度 / Hough / Otsu / 稳健拟合 / RANSAC / PCA）
   │                    python app.py   →  http://127.0.0.1:5002
   ▼
④ Y-L and other        接触角拟合（9 种模型：Young–Laplace / 圆 / 椭圆 / 多项式 / 样条 / 直线…）
   │                    python app.py   →  http://127.0.0.1:5003
   ▼
⑤ UI and output        完整流水线 + 每步自选方法 + 导出 JSON/CSV/PNG
                        python app.py   →  http://127.0.0.1:5004
```

---

## 各步骤一览

| 目录 | 步骤 | 核心内容 | 入口 |
|---|---|---|---|
| `Pre-process/` | ① 预处理 | `super_resolve`（FSRCNN / 经典回退）、`locate_drop`（联合搜索基线 + 液滴） | `app.py` :5000 |
| `Sobel-lemma/` | ② 接触线 | `detect_contact_line`：边缘响应 → 掩码轮廓带 → 接触点 → 轮廓弧 | `app.py` :5001 |
| `baseline-lemma/` | ③ 基线 | `detect_baseline`：6 种近水平直线检测算法 + 证据图 + 行剖面 | `app.py` :5002 |
| `Y-L and other methods/` | ④ 接触角 | `detect_contact_angle`：轮廓 → 基线坐标 (r,z) → 9 种拟合模型 → 左右接触角 | `app.py` :5003 |
| `UI and output/` | ⑤ 汇总 | `run_pipeline`：把 ①–④ 串起来，每步可选方法；导出 JSON/CSV/PNG、批量 CSV | `app.py` :5004 |

每个目录都包含：`README.md`（设计思路 + API）、核心算法 `.py`、
`app.py`（Flask）、`templates/index.html`（可视化页面）。

---

## 快速开始

统一依赖：`numpy opencv-python flask scipy`
（`scipy` 只被第 ④ 步的 Young–Laplace / 样条用到）

```bash
# 单独跑某一步
cd "Pre-process"            && python app.py     # http://127.0.0.1:5000
cd "Sobel-lemma"            && python app.py     # http://127.0.0.1:5001
cd "baseline-lemma"         && python app.py     # http://127.0.0.1:5002
cd "Y-L and other methods"  && python app.py     # http://127.0.0.1:5003

# 一步到位：完整流水线（推荐）
cd "UI and output"          && python app.py     # http://127.0.0.1:5004
```

在 `UI and output` 的页面里：上传图片 → 逐层选择方法 → 运行 →
查看分阶段图像与各模型角度对比 → 导出结果。

---

## 设计约定

1. **坐标系**：每步都把图像变换到「基线坐标系 `(r, z)`」——
   `r` 沿基线向右，`z` 垂直基线指向液滴内部（`z>0` 为液相一侧）。
   正立 / 倒置液滴、倾斜基线共用一套公式。
2. **接触角符号**：令接触点处切线向上为 `(t_r, t_z)`，则

   ```
   θ_left  = atan2(t_z,  t_r)
   θ_right = atan2(t_z, -t_r)
   ```

   全程不取绝对值，疏水液滴（θ>90°）也能正确表示。
3. **接口统一**：各步核心函数返回 dict，`process_*` 返回
   `{'stages': [(名称, BGR图, 描述)], 'result': ..., 'ok': bool}`，
   因此 `UI and output` 可以无缝串联。
4. **独立实现**：所有代码为重新实现，未复制 Opendrop / cadrop 源文件。

---

## 精度与限制（简）

- 第 ④ 步的 `circle` / `poly2` 与仓库既有实现 `cadrop`（CSIEC v5.3）
  在实测图上几乎逐张一致（差异 < 1°），说明几何拟合对齐；
- Young–Laplace 在合成轮廓上反演误差 < 1°；
- 整条流水线的精度**受上游制约**：掩码在台面上被平切（底部 ~3px 缺失）、
  个别图上 Hough 等算法会失败并自动回退、椭圆/样条在小窗口下易过拟合。
  这些差异正是「方法对比」要暴露的内容。
