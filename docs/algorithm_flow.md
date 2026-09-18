# 算法流程图 / Algorithm flow

下面的流程图用 Mermaid 表示，包含中文与英文节点。你可以在 GitHub 上直接查看或复制 Mermaid 文本到支持的渲染器中。

/--
示意：读取图像 → （可选）超分 → 增强 → 基线检测 → 掩码/轮廓 → 剖面点 → 左右分割与窗口选择 → 多种拟合模型尝试 → 在 z=0 处求切线并计算接触角 → 输出与可视化
---/

```mermaid
flowchart TD
  A[读取图像\nRead image] --> B{是否启用超分？\nEnable SR?}
  B -- 是 / Yes --> C[ROI 超分\nROI Super-resolution]
  B -- 否 / No --> D[图像增强（可选）\nEnhance (optional)]
  C --> D
  D --> E[基线检测\nFind substrate baseline (k,b)]
  E --> F[掩码与轮廓提取\nDroplet mask & contour]
  F --> G[剖面点提取 (r,z)\nProfile points extraction in baseline coords]
  G --> H[左右分割并选窗口\nSplit left/right & select window near contact]
  H --> I{尝试多种拟合模型\nTry multiple fit models}
  I --> I1[circle: Kasa + Gauss–Newton\n圆弧拟合]
  I --> I2[ellipse/conic: Halir–Flusser\n椭圆/二次曲线拟合]
  I --> I3[poly: quadratic r(z)\n多项式拟合]
  I --> I4[line: linear fit\n直线拟合]
  I1 --> J[在 z=0 处求切线并计算角度\nTangent at z=0 → contact angle]
  I2 --> J
  I3 --> J
  I4 --> J
  J --> K[左右角度、平均、非对称性\nLeft/right angles, mean, asymmetry]
  K --> L[置信度计算（RMS/点数/一致性）\nConfidence score (RMS/pts/agreement)]
  L --> M[输出结果与可视化（曲线/接触点/切线/基线）\nExport results & visualizations]
```

说明 / Notes

- 中文与英文节点并列，便于国内/国际读者阅读。你可以直接在 GitHub 页面上打开此文件查看 Mermaid 渲染（GitHub 原生支持 mermaid）。
- 若你想要把流程图导出为 PNG/SVG，我可以生成并把图片放到 docs/images/ 下。

