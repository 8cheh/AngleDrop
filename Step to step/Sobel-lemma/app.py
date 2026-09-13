"""Sobel-lemma 可视化页面 —— 上传图片，选择边缘算法模式检测接触线。

用法：
    python app.py                  # http://127.0.0.1:5001
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

# 复用 Pre-process 的读取 / 超分 / 定位能力（同仓库第 1 步）
_PRE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pre-process'))
if _PRE not in sys.path:
    sys.path.insert(0, _PRE)

import preprocess as pp  # noqa: E402
from sobel_lemma import (EDGE_MODES, THRESHOLD_MODES,  # noqa: E402
                         detect_contact_line, process_contact_line)

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
    return render_template('index.html', modes=EDGE_MODES, thresholds=THRESHOLD_MODES)


@app.route('/api/meta')
def api_meta():
    return jsonify({
        'sr_available': pp.sr_available(),
        'modes': EDGE_MODES,
        'thresholds': THRESHOLD_MODES,
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
    mode = str(data.get('mode', 'sobel'))
    threshold = str(data.get('threshold', 'otsu'))
    contact_band = float(data.get('contact_band', 10.0))

    if mode not in EDGE_MODES:
        return jsonify({'error': f'未知模式 {mode}'}), 400
    if threshold not in THRESHOLD_MODES:
        return jsonify({'error': f'未知阈值方式 {threshold}'}), 400

    out = process_contact_line(item['bgr'], use_sr=use_sr, sr_scale=sr_scale,
                               mode=mode, threshold=threshold,
                               contact_band=contact_band)

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
        'mode': mode,
        'mode_label': EDGE_MODES.get(mode, mode),
        'threshold': threshold,
        'threshold_label': THRESHOLD_MODES.get(threshold, threshold),
    }

    if res.get('ok'):
        payload['left'] = res['left']
        payload['right'] = res['right']
        payload['span'] = res['span']
        payload['arc_length'] = res['arc_length']
        payload['baseline'] = res['baseline']
    else:
        payload['error'] = res.get('error', '接触线检测失败')
    return jsonify(payload)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5001)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()

    print(f'Sobel-lemma UI -> http://{args.host}:{args.port}')
    print('上传图片 -> 预处理(超分可选) -> 选择边缘算法 -> 接触线检测')
    print('可用模式：', ' / '.join(EDGE_MODES))
    if not pp.sr_available():
        print('提示：未检测到 cv2.dnn_superres（需 opencv-contrib-python），超分会走经典回退')
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
