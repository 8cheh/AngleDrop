# 悬滴法界面张力 / Pendant-drop interfacial tension

本模块为 AngleDrop 增加**悬滴法界面张力（IFT）**测量能力：从一张悬滴照片拟合
Young-Laplace 形状，得到 Bond 数、顶点曲率半径与界面张力。

This adds pendant-drop interfacial tension measurement to AngleDrop: fit a
Young-Laplace shape to a hanging-drop profile and recover the Bond number, apex
radius of curvature and surface tension.

> **状态 / Status**：算法核心已完成并在**合成数据**上端到端验证。
> 尚未接入图像采集界面与针头检测 UI；尚未用真实实验数据验证。
> The numerical core is complete and validated end-to-end on synthetic data.
> Image-acquisition UI, needle detection and real-experiment validation are
> not done yet.

---

## 1. 快速开始 / Quick start

```python
from cadrop.tensiometry import (young_laplace_fit, surface_tension,
                                pixel_scale_from_needle, worthington_number,
                                AIR_DENSITY)

# (x, y) profile of a pendant drop, in image pixels, y downward
# 悬滴轮廓点，图像像素坐标，y 向下
res = young_laplace_fit(profile_xy)
print(res.to_dict())

# the needle is the ruler: 1.5 mm outer diameter spanning 300 px
px_mm = pixel_scale_from_needle(needle_diameter_mm=1.5, needle_diameter_px=300.0)

gamma = surface_tension(delta_rho=997.0 - AIR_DENSITY,
                        radius_px=res.radius_px,
                        px_size_mm=px_mm,
                        bond=res.bond)          # mN/m
```

条件数诊断 / conditioning diagnostic, on images you already have:

```bash
python shape_parameter_scan.py --dir "D:\path\to\images" --csv ps.csv
```

---

## 2. 物理 / The physics

轴对称液滴的界面满足 Young-Laplace 方程。以顶点曲率半径 `R0` 为单位无量纲化，
`s` 为从顶点起的弧长，`φ` 为切线角：

```
dr/ds   = cos(φ)
dz/ds   = sin(φ)
dφ/ds   = 2 + σ·Bo·z − sin(φ)/r
```

其中 `z` 从顶点指向液体内部，`σ = −1`（悬滴）或 `+1`（固着滴），

```
Bo = Δρ · g · R0² / γ                     （Bond 数）
```

拟合得到 `Bo` 与 `R0`（像素）后：

```
γ = Δρ · g · R0_米² / Bo
```

**因此 γ ∝ 尺度²**：像素标定错 1%，界面张力错 2%。这是整条链路里最大的单项误差源。

`Bo = 0` 时方程有精确解——单位球 `r = sin s, z = 1 − cos s, φ = s`——这是本模块最
有力的正确性测试。

---

## 3. 模块 / What is implemented

| 文件 | 内容 |
|---|---|
| `cadrop/younglaplace.py` | Young-Laplace 方程求解器，含 Bond 数灵敏度方程、体积/表面积、最近点投影 |
| `cadrop/tensiometry.py` | 五参数最小二乘拟合、尺度标定、γ、Worthington 数、形状参数 P_s |
| `shape_parameter_scan.py` | 命令行工具：扫描一个图片目录，回答「这些液滴能不能测界面张力」 |
| `tests/test_younglaplace.py` | 求解器测试（9 项） |
| `tests/test_tensiometry.py` | 拟合与派生量测试（15 项） |

### 3.1 拟合参数 / Fitted parameters

`(Bo, R0_px, apex_x, apex_y, rotation_deg)` —— 与公开文献一致的五参数。
`rotation` 表示相机不完全水平，**不要硬编码重力方向**。

目标函数是每个测量点到理论曲线的**最小法向距离**，用 KD-tree 加速。
拟合先用底部球冠做圆拟合得到顶点与 `R0`，再对 `Bo` 做粗网格扫描，最后有界最小二乘。
粗扫描是必要的：`Bo` 只通过形状进入目标函数，局部优化从错误的分支出发会收敛到
一个近似球形、残差看起来还能接受的解。

### 3.2 两个必须一起看的派生量 / Two derived quantities that matter as much as the fit

**形状参数 P_s** —— 投影面积落在顶点内切圆之外的比例，球形为 0，重力变形越大越高。
它**与尺度和液体都无关**，所以可以直接在已有图片上计算。公开临界值（误差目标
0.1 mJ/m²）为 0.19–0.35（视构型）。**P_s 低于阈值时，界面张力不是「测得不准」，
而是「形状里根本没有这个信息」。**

**Worthington 数 Wo** —— `Wo = Δρ·g·V/(π·γ·D_针)`，刻画液滴接近脱落的程度。
注意文献中有两套约定相差 `2π`，比较阈值前必须确认用的哪一套；本模块用
Kratz & Kierfeld / Berry 约定。

---

## 4. 验证 / Validation

全部为**合成数据**，不需要任何实验照片。

| 测试 | 结果 |
|---|---|
| `Bo = 0` 精确退化单位球 | 误差 < 1e-8 |
| 顶点奇点 `(dφ/ds)(0) = 1` | 通过（L'Hôpital） |
| 体积/表面积（单位球赤道） | `2π/3` 与 `2π`，误差 < 1e-6 |
| Bond 数灵敏度 vs 有限差分 | 一致（rtol 1e-4） |
| 最近点投影自洽 | 通过 |
| 无噪声回环：`Bo → 轮廓 → 拟合 → Bo` | 误差 < 0.5%，RMS 达到目标函数下界 |
| 0.5 px 噪声回环 | Bo 误差 < 15%（4 个噪声种子） |
| 相机倾斜 3° 回环 | 角度恢复误差 < 1.5° |
| **端到端 γ 恢复（含水/空气真实参数）** | 72.00 → 误差 < 5% |
| 1% 尺度误差 → 2.02% γ 误差 | 精确 |
| `P_s` 数据驱动版 vs 精确版 | 一致到 ~2% |

**独立交叉验证**：本求解器复现了公开的 selected-plane 经验关系
`Bo = 0.1756x² + 0.5234x³ − 0.2563x⁴`（`x` 为 `z = 2R0` 处的半径除以 `R0`），
在 `Bo = 0.05–0.40` 范围内误差 **< 0.4%**。这一条同时验证了方程、符号约定与顶点处理。

---

## 5. 已知限制 / Known limitations

1. **Bo 上限约 0.6。** 超过之后子午线不再达到竖直切线，轮廓退化为半无限柱而非
   紧凑液滴。悬滴张力测量的实际工作区间是 `Bo ≈ 0.1–0.5`（Dekker et al. 报告
   0.356），所以实用范围内没问题，但更大的液滴需要更高阶的解分支，本模块未实现。
2. **只做了悬滴（pendant）。** 固着滴的 `σ = +1` 在求解器里已支持，但拟合器与顶点
   检测仍假定顶点在最低点。
3. **未接入界面。** 没有针头检测、没有标定交互页面、没有 GUI 集成。
4. **未用真实数据验证。** 合成数据能证明代码正确，不能证明实验可行。
5. **依赖 scipy。** `cadrop/__init__.py` 未改动，所以既有的 `import cadrop` 不受影响；
   新功能需显式 `from cadrop.tensiometry import ...`。

---

## 6. 硬件要点 / Hardware requirements

详见仓库根目录的 `界面张力-技术方案.md`。核心几条：

| 项目 | 建议 | 理由 |
|---|---|---|
| 针头外径 | 1.0–1.5 mm（追求精度用 3–6 mm 毛细管） | Bo 条件数；Saad 明确说细针「一般不是最优选择」 |
| 针头像素宽 | ≥ 300 px | 尺度误差 0.13% → γ 误差 0.27% |
| 放大率 | 110–140 px/mm | 液滴高度跨 800 px |
| 光源 | 漫射背光 + **单色（蓝光）滤光片** | 单色对边缘检测的改善是分辨率翻倍的 4 倍 |
| 标定 | **与测量同光路同介质** | 否则引入 5–9% 系统误差，比其余项加起来还大 |

---

## 7. 引用 / References

本模块是**独立实现**，代码不来自 OpenDrop（GPL-3.0）。数学依据为公开文献：

- Bashforth & Adams, *An Attempt to Test the Theories of Capillary Action* (1883)
- Rotenberg, Boruvka & Neumann, *J. Colloid Interface Sci.* **93** (1983) 169
- del Río & Neumann, *J. Colloid Interface Sci.* **196** (1997) 136
- Berry, Neeson, Dagastine, Chan & Tabor, *J. Colloid Interface Sci.* **454** (2015) 226
- Huang et al., *J. Open Source Software* **6**(58) (2021) 2604
- Hoorfar, PhD thesis, University of Toronto (2006) — 形状参数 P_s 与临界值
- Pan & Trusler, *Int. J. Thermophys.* **43** (2022) 65 — γ ∝ 尺度² 与光路折射修正
