"""Y-L and other methods 可视化页面 —— 选择拟合方式计算接触角。

用法：
    python app.py                  # http://127.0.0.1:5003
    python app.py --port 8000
"""
import argparse
import base64
import os
import sys
import uuid
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# 复用同仓库前面几步：Pre-process / Sobel-lemma
_HERE = os.path.dirname(os.path.abspath(__file__))
_PRE = os.path.abspath(os.path.join(_HERE, '..', 'Pre-process'))
_SOBEL = os.path.abspath(os.path.join(_HERE, '..', 'Sobel-lemma'))
for _p in (_PRE, _SOBEL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preprocess as pp  # noqa: E402
from contact_angle import (ANGLE_METHODS,  # noqa: E402
                           process_contact_angle)

try:
    from sobel_lemma import EDGE_MODES, THRESHOLD_MODES, detect_contact_line  # noqa: E402
    _HAVE_SOBEL = True
except Exception:  # pragma: no cover
    EDGE_MODES, THRESHOLD_MODES = {}, {}
    _HAVE_SOBEL = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024   # 256 MB

IMAGES: OrderedDict = OrderedDict()   # id -> {'name', 'bgr'}


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


@app.route('/')
def index():
    return render_template('index.html', methods=ANGLE_METHODS,
                           edge_modes=EDGE_MODES, thresholds=THRESHOLD_MODES)


@app.route('/api/meta')
def api_meta():
    return jsonify({
        'sr_available': pp.sr_available(),
        'methods': ANGLE_METHODS,
        'edge_modes': EDGE_MODES,
        'thresholds': THRESHOLD_MODES,
        'have_sobel': _HAVE_SOBEL,
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
    return jsonify([{'id': k, 'name': v['name']} for k, v in IMAGES.items()])


@app.route('/api/clear', methods=['POST'])
def api_clear():
    IMAGES.clear()
    return jsonify({'ok': True})


@app.route('/api/clear_upload', methods=['POST'])
def api_clear_upload():
    data = request.get_json(silent=True) or {}
    uid = data.get('id', '')
    if uid in IMAGES:
        del IMAGES[uid]
        return jsonify({'ok': True})
    return jsonify({'ok': False}), 404


@app.route('/api/process', methods=['POST'])
def api_process():
    data = request.get_json(silent=True) or {}
    uid = data.get('id', '')
    item = IMAGES.get(uid)
    if item is None:
        return jsonify({'error': 'image not found'}), 404

    use_sr = bool(data.get('use_sr', False))
    sr_scale = float(data.get('sr_scale', 3.0))
    method = str(data.get('method', 'young_laplace'))
    window_frac = float(data.get('window_frac', 0.25))
    compare = bool(data.get('compare', True))
    edge_mode = str(data.get('edge_mode', 'sobel'))
    threshold = str(data.get('threshold', 'otsu'))

    if method not in ANGLE_METHODS:
        return jsonify({'error': f'未知方法 {method}'}), 400

    def contact_fn(work, loc):
        if _HAVE_SOBEL and loc.get('ok') and edge_mode in EDGE_MODES:
            cl = detect_contact_line(work, loc, mode=edge_mode, threshold=threshold)
            if cl.get('ok'):
                return cl
        return None

    out = process_contact_angle(item['bgr'], use_sr=use_sr, sr_scale=sr_scale,
                                method=method, window_frac=window_frac,
                                compare=compare, contact_fn=contact_fn)
    contact_used = any(s[0].startswith('接触点') for s in out['stages'])

    stages = []
    for name, img, desc in out['stages']:
        stages.append({
            'name': name,
            'desc': desc,
            'url': _encode(_limit(img)),
            'w': img.shape[1],
            'h': img.shape[0],
        })

    res = out['result']
    payload = {
        'name': item['name'],
        'sr_active': out['sr_active'],
        'ok': out['ok'],
        'stages': stages,
        'method': res.get('method', method),
        'method_label': res.get('method_label', ANGLE_METHODS.get(method, method)),
        'requested_method': method,
    }

    if res.get('ok'):
        payload['left'] = {'angle_deg': res['left']['angle_deg'],
                           'rms_px': res['left']['rms_px'],
                           'n_points': res['left']['n_points']}
        payload['right'] = {'angle_deg': res['right']['angle_deg'],
                            'rms_px': res['right']['rms_px'],
                            'n_points': res['right']['n_points']}
        payload['avg_angle'] = res['avg_angle']
        payload['asymmetry'] = res['asymmetry']
        payload['confidence'] = res['confidence']
        payload['base_width'] = res['base_width']
        payload['height'] = res['height']
        payload['baseline'] = res['baseline']
        payload['contact_source'] = ('sobel_lemma' if contact_used else 'internal')
        payload['comparison'] = res.get('comparison', [])
        if res['left'].get('model') and res['left']['model'].get('kind') == 'yl':
            payload['yl'] = {'b_left': res['left']['model']['b'],
                             'lam_left': res['left']['model']['lam'],
                             'b_right': res['right']['model']['b'] if res['right'].get('model') else None,
                             'lam_right': res['right']['model']['lam'] if res['right'].get('model') else None}
    else:
        payload['error'] = res.get('error', '接触角拟合失败')
    return jsonify(payload)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5003)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()

    print(f'Y-L and other methods UI -> http://{args.host}:{args.port}')
    print('上传图片 -> 预处理 -> (接触线) -> 选择拟合方式 -> 接触角')
    print('可用拟合方法：')
    for k, v in ANGLE_METHODS.items():
        print(f'  {k:18s} {v}')
    if not pp.sr_available():
        print('提示：未检测到 cv2.dnn_superres（需 opencv-contrib-python），超分会走经典回退')
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
