"""UI and output · 完整流水线编排（Step to step 的总结）。

把前面每一步串成一条可配置的流水线，每一步都能自选方法：

    第 1 步 Pre-process      超分辨率（可选）+ 液滴定位
    第 2 步 baseline-lemma   台面基线识别（6 种算法，可选）
    第 3 步 Sobel-lemma      接触线 / 接触点检测（6 种边缘 × 3 种阈值）
    第 4 步 Y-L and other    接触角拟合（9 种模型）

本模块只做编排与结果整理，不做具体算法（各步算法在各自目录里）。
输出统一为：

    {
      'ok': bool,
      'config': {...},                 # 本次使用的全部方法/参数
      'steps': {                       # 每一步的结构化结果
          'preprocess': {...},
          'baseline':   {...},
          'contact':    {...},
          'angle':      {...},
      },
      'stages': [(step, name, bgr, desc), ...],   # 各阶段图，供 UI 展示
      'error': str | None,
    }
"""
from __future__ import annotations

import csv
import io
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 把前四步加入 import 路径
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
for _name in ('Pre-process', 'Sobel-lemma', 'baseline-lemma', 'Y-L and other methods'):
    _p = os.path.join(_ROOT, _name)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preprocess as pp                    # noqa: E402  第 1 步
import baseline_lemma as bl                # noqa: E402  第 2 步
import sobel_lemma as sl                   # noqa: E402  第 3 步
import contact_angle as ca                 # noqa: E402  第 4 步

__all__ = ['run_pipeline', 'result_to_dict', 'result_to_csv', 'METHOD_CATALOG']

# 每一步可选方法的总目录（供 UI 直接渲染下拉框）
METHOD_CATALOG = {
    'baseline': dict(bl.BASELINE_MODES),
    'edge': dict(sl.EDGE_MODES),
    'threshold': dict(sl.THRESHOLD_MODES),
    'angle': dict(ca.ANGLE_METHODS),
}


def _rebuild_loc(work: np.ndarray, k: float, b: float, invert: bool,
                 clearance: float = 3.0) -> Optional[Dict]:
    """用给定基线重新分割液滴，返回与基线一致的 loc（掩码 + 包围盒）。"""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    seg = getattr(pp, '_segment', None)
    if seg is None:
        return None
    try:
        mask, score = seg(gray, float(k), float(b), bool(invert), clearance=clearance)
    except Exception:
        return None
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return {
        'ok': True, 'mask': mask, 'k': float(k), 'b': float(b),
        'invert': bool(invert),
        'bbox': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        'area': int(mask.sum() // 255), 'score': float(score),
    }


def run_pipeline(bgr: np.ndarray, *,
                 use_sr: bool = False,
                 sr_scale: float = 3.0,
                 baseline_mode: str = 'preprocess',
                 max_slope: float = 0.2,
                 band: int = 60,
                 edge_mode: str = 'sobel',
                 threshold: str = 'otsu',
                 contact_band: float = 10.0,
                 angle_method: str = 'young_laplace',
                 window_frac: float = 0.25,
                 v_frac: float = 1.0,
                 compare: bool = True,
                 seed: int = 0) -> Dict:
    """跑完整流水线。参数与各步一一对应。

    baseline_mode：
        'preprocess' —— 直接用第 1 步内置基线（不再单独识别）；
        其它值       —— 用第 2 步 baseline-lemma 的对应算法重新识别基线，
                        并用新基线重新分割液滴，保持后续步骤一致。
    """
    stages: List[Tuple[int, str, np.ndarray, str]] = []
    cfg = {
        'use_sr': bool(use_sr), 'sr_scale': float(sr_scale),
        'baseline_mode': baseline_mode, 'max_slope': float(max_slope),
        'band': int(band), 'edge_mode': edge_mode, 'threshold': threshold,
        'contact_band': float(contact_band), 'angle_method': angle_method,
        'window_frac': float(window_frac), 'v_frac': float(v_frac),
        'compare': bool(compare),
    }

    # ---------------- 第 1 步：预处理（超分 + 定位） ----------------
    h, w = bgr.shape[:2]
    stages.append((1, '原图', bgr.copy(), f'{w}×{h}'))

    work = bgr
    sr_active = False
    if use_sr and sr_scale > 1:
        sr_active = pp.sr_available()
        work = pp.super_resolve(work, sr_scale)
        tag = '超分辨率' if sr_active else '超分辨率（经典回退）'
        stages.append((1, tag, work.copy(),
                       f'{work.shape[1]}×{work.shape[0]}，scale={int(round(sr_scale))}'))

    loc = pp.locate_drop(work)
    stages.append((1, '液滴定位（预处理）', pp.annotate(work, loc),
                   ('bbox ' + str(loc['bbox'])) if loc.get('ok') else '定位失败'))
    preprocess_step = {
        'ok': bool(loc.get('ok')),
        'sr_active': sr_active,
        'bbox': loc.get('bbox'),
        'area': loc.get('area'),
        'baseline': ({'k': float(loc['k']), 'b': float(loc['b']),
                      'invert': bool(loc['invert'])} if loc.get('ok') else None),
        'error': None if loc.get('ok') else loc.get('error', '未找到液滴'),
    }
    if not loc.get('ok'):
        return {'ok': False, 'config': cfg, 'steps': {'preprocess': preprocess_step},
                'stages': stages, 'error': preprocess_step['error']}

    # ---------------- 第 2 步：基线识别（可选） ----------------
    baseline_step = {
        'ok': True, 'mode': 'preprocess', 'mode_label': '预处理内置基线',
        'k': float(loc['k']), 'b': float(loc['b']), 'invert': bool(loc['invert']),
        'rmse': None, 'quality': None, 'score': None, 'reference': None,
    }
    loc_used = loc
    if baseline_mode != 'preprocess' and baseline_mode in bl.BASELINE_MODES:
        bres = bl.detect_baseline(work, loc, mode=baseline_mode,
                                  max_slope=max_slope, band=band, seed=seed)
        if bres.get('ok'):
            baseline_step = {
                'ok': True, 'mode': baseline_mode,
                'mode_label': bres['mode_label'],
                'k': float(bres['k']), 'b': float(bres['b']),
                'invert': bool(bres['invert']),
                'rmse': bres.get('rmse'), 'quality': bres.get('quality'),
                'score': bres.get('score'), 'reference': bres.get('reference'),
                'slope_deg': bres.get('slope_deg'),
                'n_inliers': bres.get('n_inliers'), 'n_points': bres.get('n_points'),
            }
            rebuilt = _rebuild_loc(work, bres['k'], bres['b'], bres['invert'])
            if rebuilt is not None:
                loc_used = rebuilt
            if bres.get('profile_vis') is not None:
                stages.append((2, '证据 · 行剖面', bres['profile_vis'],
                               '基线算法内部使用的水平边缘强度剖面'))
            stages.append((2, f'基线证据 · {bres["mode_label"]}', bres['evidence_vis'],
                           '候选点 / 候选线段（绿=内点）'))
            stages.append((2, '基线识别结果', bres['vis'],
                           f'y = {bres["k"]:+.4f}x + {bres["b"]:.1f}'))
        else:
            baseline_step = {'ok': False, 'mode': baseline_mode,
                             'mode_label': bl.BASELINE_MODES.get(baseline_mode, baseline_mode),
                             'k': float(loc['k']), 'b': float(loc['b']),
                             'invert': bool(loc['invert']),
                             'rmse': None, 'quality': None, 'score': None,
                             'reference': None, 'error': bres.get('error', '基线识别失败'),
                             'fallback': 'preprocess'}
            stages.append((2, '基线识别失败（回退预处理基线）', work.copy(),
                           bres.get('error', '')))

    # ---------------- 第 3 步：接触线 / 接触点 ----------------
    contact = None
    if edge_mode in sl.EDGE_MODES:
        cbres = sl.detect_contact_line(work, loc_used, mode=edge_mode,
                                       threshold=threshold, contact_band=contact_band)
    else:
        cbres = {'ok': False, 'error': f'未知边缘模式 {edge_mode}'}

    if cbres.get('ok'):
        contact_step = {
            'ok': True, 'mode': edge_mode,
            'mode_label': sl.EDGE_MODES.get(edge_mode, edge_mode),
            'threshold': threshold,
            'threshold_label': sl.THRESHOLD_MODES.get(threshold, threshold),
            'left': cbres['left'], 'right': cbres['right'],
            'span': cbres['span'], 'arc_length': cbres['arc_length'],
            'baseline': cbres['baseline'],
        }
        contact = cbres
        edge_vis = np.zeros((*cbres['edge'].shape, 3), dtype=np.uint8)
        edge_vis[cbres['edge'] > 0] = (120, 255, 120)
        stages.append((3, f'边缘图 · {contact_step["mode_label"]}', edge_vis,
                       f'阈值：{contact_step["threshold_label"]}'))
        stages.append((3, '接触线检测结果', cbres['vis'],
                       f"L({cbres['left']['x']:.1f},{cbres['left']['y']:.1f}) "
                       f"R({cbres['right']['x']:.1f},{cbres['right']['y']:.1f})"))
    else:
        contact_step = {'ok': False, 'mode': edge_mode,
                        'mode_label': sl.EDGE_MODES.get(edge_mode, edge_mode),
                        'threshold': threshold,
                        'error': cbres.get('error', '接触线检测失败'),
                        'fallback': 'internal'}
        stages.append((3, '接触线检测失败（回退内置接触点）', work.copy(),
                       cbres.get('error', '')))

    # ---------------- 第 4 步：接触角拟合 ----------------
    ares = ca.detect_contact_angle(work, loc_used, contact,
                                   method=angle_method, window_frac=window_frac,
                                   v_frac=v_frac, compare=compare)

    if ares.get('ok'):
        angle_step = {
            'ok': True, 'method': ares['method'],
            'method_label': ares['method_label'],
            'requested_method': angle_method,
            'left': ares['left']['angle_deg'], 'right': ares['right']['angle_deg'],
            'avg_angle': ares['avg_angle'], 'asymmetry': ares['asymmetry'],
            'confidence': ares['confidence'],
            'left_rms': ares['left']['rms_px'], 'right_rms': ares['right']['rms_px'],
            'left_points': ares['left']['n_points'], 'right_points': ares['right']['n_points'],
            'base_width': ares['base_width'], 'height': ares['height'],
            'baseline': ares['baseline'],
            'comparison': ares.get('comparison', []),
            'yl': None,
        }
        if ares['left'].get('model') and ares['left']['model'].get('kind') == 'yl':
            rm = ares['right'].get('model') or {}
            angle_step['yl'] = {
                'b_left': ares['left']['model'].get('b'),
                'lam_left': ares['left']['model'].get('lam'),
                'b_right': rm.get('b'), 'lam_right': rm.get('lam'),
            }
        stages.append((4, f'接触角拟合 · {ares["method_label"]}', ares['vis'],
                       f'平均 {ares["avg_angle"]:.2f}°'))
        if ares.get('comparison'):
            chart = ca._comparison_chart(ares['comparison'])
            if chart is not None:
                stages.append((4, '方法对比', chart, '各拟合模型平均接触角（度）'))
    else:
        angle_step = {'ok': False, 'method': angle_method,
                      'method_label': ca.ANGLE_METHODS.get(angle_method, angle_method),
                      'error': ares.get('error', '接触角拟合失败'), 'comparison': []}
        stages.append((4, '接触角拟合失败', work.copy(), ares.get('error', '')))

    steps = {
        'preprocess': preprocess_step,
        'baseline': baseline_step,
        'contact': contact_step,
        'angle': angle_step,
    }
    ok = bool(preprocess_step['ok'] and angle_step['ok'])
    return {'ok': ok, 'config': cfg, 'steps': steps, 'stages': stages,
            'image_size': [int(bgr.shape[1]), int(bgr.shape[0])],
            'error': None if ok else angle_step.get('error')}


# --------------------------------------------------------------------------- #
# 导出：JSON / CSV
# --------------------------------------------------------------------------- #
def result_to_dict(result: Dict, name: str = '') -> Dict:
    """整理成可 JSON 序列化的精简结果。"""
    s = result.get('steps', {})
    pre, base, con, ang = (s.get('preprocess', {}), s.get('baseline', {}),
                           s.get('contact', {}), s.get('angle', {}))
    out = {
        'name': name,
        'ok': bool(result.get('ok')),
        'image_size': result.get('image_size'),
        'config': result.get('config', {}),
        'preprocess': {
            'ok': pre.get('ok'), 'sr_active': pre.get('sr_active'),
            'bbox': pre.get('bbox'), 'area': pre.get('area'),
            'baseline': pre.get('baseline'),
        },
        'baseline': {
            'ok': base.get('ok'), 'mode': base.get('mode'),
            'mode_label': base.get('mode_label'),
            'k': base.get('k'), 'b': base.get('b'), 'invert': base.get('invert'),
            'rmse': base.get('rmse'), 'quality': base.get('quality'),
            'score': base.get('score'), 'n_inliers': base.get('n_inliers'),
        },
        'contact': {
            'ok': con.get('ok'), 'mode': con.get('mode'),
            'mode_label': con.get('mode_label'), 'threshold': con.get('threshold'),
            'left': con.get('left'), 'right': con.get('right'),
            'span': con.get('span'), 'arc_length': con.get('arc_length'),
        },
        'angle': {
            'ok': ang.get('ok'), 'method': ang.get('method'),
            'method_label': ang.get('method_label'),
            'left': ang.get('left'), 'right': ang.get('right'),
            'avg': ang.get('avg_angle'), 'asymmetry': ang.get('asymmetry'),
            'confidence': ang.get('confidence'),
            'left_rms': ang.get('left_rms'), 'right_rms': ang.get('right_rms'),
            'base_width': ang.get('base_width'), 'height': ang.get('height'),
            'yl': ang.get('yl'),
            'comparison': ang.get('comparison', []),
        },
        'error': result.get('error'),
    }
    return out


CSV_COLUMNS = [
    'name', 'ok', 'sr_active', 'baseline_mode', 'edge_mode', 'threshold',
    'angle_method', 'angle_left', 'angle_right', 'angle_avg', 'asymmetry',
    'confidence', 'base_width', 'height', 'baseline_k', 'baseline_b',
    'contact_left_x', 'contact_left_y', 'contact_right_x', 'contact_right_y',
    'span', 'arc_length', 'error',
]


def result_to_csv(result: Dict, name: str = '') -> str:
    """把结果整理成单行 CSV（含表头），便于批量汇总。"""
    d = result_to_dict(result, name)
    pre, base, con, ang = (d['preprocess'], d['baseline'], d['contact'], d['angle'])
    left = con.get('left') or {}
    right = con.get('right') or {}
    row = {
        'name': name, 'ok': int(d['ok']),
        'sr_active': int(bool(pre.get('sr_active'))),
        'baseline_mode': base.get('mode'),
        'edge_mode': con.get('mode'), 'threshold': con.get('threshold'),
        'angle_method': ang.get('method'),
        'angle_left': ang.get('left'), 'angle_right': ang.get('right'),
        'angle_avg': ang.get('avg'), 'asymmetry': ang.get('asymmetry'),
        'confidence': ang.get('confidence'),
        'base_width': ang.get('base_width'), 'height': ang.get('height'),
        'baseline_k': base.get('k'), 'baseline_b': base.get('b'),
        'contact_left_x': left.get('x'), 'contact_left_y': left.get('y'),
        'contact_right_x': right.get('x'), 'contact_right_y': right.get('y'),
        'span': con.get('span'), 'arc_length': con.get('arc_length'),
        'error': d.get('error') or '',
    }
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    w.writerow({k: ('' if row.get(k) is None else row.get(k)) for k in CSV_COLUMNS})
    return buf.getvalue()
