"""UI and output · 完整流水线可视化页面（Step to step 总结）。

把 Step to step 的四步串成一个界面，每一步都能自选方法：

    上传 → ① 预处理 → ② 基线识别 → ③ 接触线 → ④ 接触角 → 导出

用法：
    python app.py                  # http://127.0.0.1:5004
    python app.py --port 8000
"""
import argparse
import base64
import os
import sys
import time
import uuid
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import pipeline as P

_HERE = os.path.dirname(os.path.abspath(__file__))
_PRE = os.path.abspath(os.path.join(_HERE, '..', 'Pre-process'))
if _PRE not in sys.path:
    sys.path.insert(0, _PRE)
import preprocess as pp  # noqa: E402

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024   # 512 MB

IMAGES: OrderedDict = OrderedDict()      # id -> {'name', 'bgr'}
LAST_RESULTS: OrderedDict = OrderedDict()  # id -> {'name','steps','annotated','time'}


def _final_image(result: dict):
    """取流程中最后一张「接触角拟合」图作为标注图（用于导出）。"""
    chosen = None
    for step, sname, img, _desc in result['stages']:
        if step == 4 and '\u62df\u5408' in sname:
            chosen = img
    if chosen is None and result['stages']:
        chosen = result['stages'][-1][2]
    return chosen


def _store_result(uid: str, name: str, result: dict, dt: float = 0.0):
    """只保留导出所需的精简结果 + 一张标注图，避免累积全部分阶段大图。"""
    LAST_RESULTS[uid] = {
        'name': name,
        'trimmed': {
            'ok': result.get('ok'), 'steps': result.get('steps', {}),
            'config': result.get('config', {}),
            'image_size': result.get('image_size'),
            'error': result.get('error'),
        },
        'annotated': _final_image(result),
        'time': dt,
    }


def _decode(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode(bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode() if ok else ''


def _limit(bgr: np.ndarray, max_side: int = 1400) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    s = max_side / max(h, w)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _config_from(data: dict) -> dict:
    return {
        'use_sr': bool(data.get('use_sr', False)),
        'sr_scale': float(data.get('sr_scale', 3.0)),
        'baseline_mode': str(data.get('baseline_mode', 'preprocess')),
        'max_slope': float(data.get('max_slope', 0.2)),
        'band': int(data.get('band', 60)),
        'edge_mode': str(data.get('edge_mode', 'sobel')),
        'threshold': str(data.get('threshold', 'otsu')),
        'contact_band': float(data.get('contact_band', 10.0)),
        'angle_method': str(data.get('angle_method', 'young_laplace')),
        'window_frac': float(data.get('window_frac', 0.25)),
        'compare': bool(data.get('compare', True)),
    }


def _stages_payload(result: dict):
    out = []
    for step, name, img, desc in result['stages']:
        out.append({
            'step': int(step), 'name': name, 'desc': desc,
            'url': _encode(_limit(img)),
            'w': int(img.shape[1]), 'h': int(img.shape[0]),
        })
    return out


@app.route('/')
def index():
    return render_template('index.html',
                           baseline_modes=P.METHOD_CATALOG['baseline'],
                           edge_modes=P.METHOD_CATALOG['edge'],
                           thresholds=P.METHOD_CATALOG['threshold'],
                           angle_methods=P.METHOD_CATALOG['angle'])


@app.route('/api/meta')
def api_meta():
    return jsonify({
        'sr_available': pp.sr_available(),
        'catalog': P.METHOD_CATALOG,
    })


@app.route('/api/upload', methods=['POST'])
def api_upload():
    files = request.files.getlist('files')
    added = []
    for f in files:
        if not f or not f.filename:
            continue
        bgr = _decode(f.read())
        if bgr is None:
            continue
        uid = uuid.uuid4().hex[:12]
        IMAGES[uid] = {'name': f.filename, 'bgr': bgr}
        added.append({'id': uid, 'name': f.filename})
    return jsonify({'added': added, 'total': len(IMAGES)})


@app.route('/api/images')
def api_images():
    return jsonify([
        {'id': k, 'name': v['name'],
         'has_result': k in LAST_RESULTS}
        for k, v in IMAGES.items()
    ])


@app.route('/api/clear', methods=['POST'])
def api_clear():
    IMAGES.clear()
    LAST_RESULTS.clear()
    return jsonify({'ok': True})


@app.route('/api/clear_upload', methods=['POST'])
def api_clear_upload():
    data = request.get_json(silent=True) or {}
    uid = data.get('id', '')
    if uid in IMAGES:
        del IMAGES[uid]
        LAST_RESULTS.pop(uid, None)
        return jsonify({'ok': True})
    return jsonify({'ok': False}), 404


@app.route('/api/process', methods=['POST'])
def api_process():
    data = request.get_json(silent=True) or {}
    uid = data.get('id', '')
    item = IMAGES.get(uid)
    if item is None:
        return jsonify({'error': 'image not found'}), 404

    cfg = _config_from(data)
    t0 = time.time()
    try:
        result = P.run_pipeline(item['bgr'], **cfg)
    except Exception as e:  # 防止某一步异常直接 500
        return jsonify({'error': f'流水线异常：{e}'}), 500
    dt = time.time() - t0
    _store_result(uid, item['name'], result, dt)

    steps = result['steps']
    payload = {
        'name': item['name'],
        'ok': result['ok'],
        'elapsed': round(dt, 2),
        'config': cfg,
        'stages': _stages_payload(result),
        'preprocess': steps['preprocess'],
        'baseline': steps['baseline'],
        'contact': steps['contact'],
        'angle': steps['angle'],
        'error': result.get('error'),
    }
    return jsonify(payload)


@app.route('/api/export')
def api_export():
    uid = request.args.get('id', '')
    fmt = request.args.get('fmt', 'json').lower()
    rec = LAST_RESULTS.get(uid)
    if rec is None:
        return jsonify({'error': 'no result for this image, run first'}), 404
    name = os.path.splitext(rec['name'])[0]
    if fmt == 'csv':
        txt = P.result_to_csv(rec['trimmed'], rec['name'])
        return Response(txt, mimetype='text/csv',
                        headers={'Content-Disposition': f'attachment; filename="{name}_result.csv"'})
    import json as _json
    txt = _json.dumps(P.result_to_dict(rec['trimmed'], rec['name']),
                      ensure_ascii=False, indent=2)
    return Response(txt, mimetype='application/json',
                    headers={'Content-Disposition': f'attachment; filename="{name}_result.json"'})


@app.route('/api/export_image')
def api_export_image():
    uid = request.args.get('id', '')
    rec = LAST_RESULTS.get(uid)
    if rec is None:
        return jsonify({'error': 'no result for this image, run first'}), 404
    # 直接使用流程中保留的标注图
    chosen = rec.get('annotated')
    if chosen is None:
        return jsonify({'error': 'no image'}), 404
    ok, buf = cv2.imencode('.png', chosen)
    name = os.path.splitext(rec['name'])[0]
    return Response(buf.tobytes(), mimetype='image/png',
                    headers={'Content-Disposition': f'attachment; filename="{name}_annotated.png"'})


@app.route('/api/batch', methods=['POST'])
def api_batch():
    """用同一套方法批量跑所有已上传图片，返回 CSV。"""
    data = request.get_json(silent=True) or {}
    cfg = _config_from(data)
    if not IMAGES:
        return jsonify({'error': '没有已上传的图片'}), 400
    rows = [','.join(P.CSV_COLUMNS)]
    summary = []
    for uid, item in IMAGES.items():
        try:
            result = P.run_pipeline(item['bgr'], **cfg)
        except Exception as e:
            summary.append({'id': uid, 'name': item['name'], 'ok': False, 'error': str(e)})
            continue
        _store_result(uid, item['name'], result)
        csv = P.result_to_csv(result, item['name']).splitlines()
        if len(csv) >= 2:
            rows.append(csv[1])
        ang = result['steps']['angle']
        summary.append({'id': uid, 'name': item['name'], 'ok': bool(result['ok']),
                        'angle': ang.get('avg_angle'), 'method': ang.get('method_label')})
    return jsonify({'csv': '\n'.join(rows) + '\n', 'summary': summary,
                    'count': len(summary)})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5004)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()

    print(f'UI and output · 完整流水线 -> http://{args.host}:{args.port}')
    print('上传 -> ① 预处理 -> ② 基线 -> ③ 接触线 -> ④ 接触角 -> 导出')
    print('  basline :', ' / '.join(P.METHOD_CATALOG['baseline']))
    print('  edge    :', ' / '.join(P.METHOD_CATALOG['edge']))
    print('  angle   :', ' / '.join(P.METHOD_CATALOG['angle']))
    if not pp.sr_available():
        print('提示：未检测到 cv2.dnn_superres（需 opencv-contrib-python），超分会走经典回退')
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
