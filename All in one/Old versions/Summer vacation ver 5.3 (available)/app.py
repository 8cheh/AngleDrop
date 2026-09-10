"""cadrop web interface — upload-based, with a snipping-tool style region selector.

Usage:
    python app.py                      # empty; upload images in the browser
    python app.py --port 8000
    python app.py --image-dir <folder> # optional: pre-load a folder (still uploadable)
"""
import argparse
import base64
import io
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from cadrop import METHODS, measure, measure_roi
from cadrop.detect import read_image
from cadrop.render import annotate

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024   # 256 MB per request

# In-memory image store: id -> {'name': str, 'bgr': np.ndarray}
IMAGES: OrderedDict = OrderedDict()
CONFIG = {'method': 'auto', 'use_enhance': False}


def _decode(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _add_image(name: str, bgr: np.ndarray) -> str:
    uid = uuid.uuid4().hex[:12]
    IMAGES[uid] = {'name': name, 'bgr': bgr}
    return uid


def _encode_jpeg(bgr: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode() if ok else ''


def _encode_thumb(bgr: np.ndarray, width: int = 120) -> str:
    h, w = bgr.shape[:2]
    scale = width / w
    small = cv2.resize(bgr, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return _encode_jpeg(small, 80)


def _region(data):
    r = data.get('region')
    if not isinstance(r, dict):
        return None
    try:
        return (float(r['x']), float(r['y']), float(r['w']), float(r['h']))
    except (KeyError, TypeError, ValueError):
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/images')
def api_images():
    return jsonify([{'id': k, 'name': v['name']} for k, v in IMAGES.items()])


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
        uid = _add_image(f.filename, bgr)
        added.append({'id': uid, 'name': f.filename})
    return jsonify({'added': added, 'total': len(IMAGES)})


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


@app.route('/api/image', methods=['GET'])
def api_image():
    uid = request.args.get('id', '')
    item = IMAGES.get(uid)
    if item is None:
        return jsonify({'error': 'image not found'}), 404
    img = item['bgr']
    return jsonify({'url': _encode_jpeg(img), 'size': list(img.shape[:2]),
                    'name': item['name']})


@app.route('/api/thumb', methods=['GET'])
def api_thumb():
    uid = request.args.get('id', '')
    item = IMAGES.get(uid)
    if item is None:
        return jsonify({'error': 'image not found'}), 404
    return jsonify({'url': _encode_thumb(item['bgr'])})


@app.route('/api/measure', methods=['POST'])
def api_measure():
    data = request.get_json(silent=True) or {}
    uid = data.get('id', '')
    item = IMAGES.get(uid)
    if item is None:
        return jsonify({'error': 'image not found'}), 404
    method = data.get('method', CONFIG['method'])
    use_enhance = bool(data.get('use_enhance', CONFIG['use_enhance']))
    region = _region(data)

    t0 = time.time()
    if region is not None:
        res = measure_roi(item['bgr'], region, item['name'],
                          method=method, use_enhance=use_enhance)
    else:
        res = measure(item['bgr'], item['name'], method=method, use_enhance=use_enhance)
    dt = time.time() - t0

    payload = res.to_dict()
    payload['elapsed_ms'] = round(dt * 1000, 1)
    if res.ok:
        payload['annotated'] = _encode_jpeg(annotate(item['bgr'], res))
    else:
        payload['annotated'] = _encode_jpeg(item['bgr'])
    return jsonify(payload)


@app.route('/api/run_all', methods=['POST'])
def api_run_all():
    data = request.get_json(silent=True) or {}
    method = data.get('method', CONFIG['method'])
    use_enhance = bool(data.get('use_enhance', CONFIG['use_enhance']))

    results = []
    for uid, item in IMAGES.items():
        res = measure(item['bgr'], item['name'], method=method, use_enhance=use_enhance)
        d = res.to_dict()
        d['id'] = uid
        d.pop('image_size', None)
        results.append(d)
    return jsonify({'count': len(results), 'results': results})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--image-dir', default=None,
                    help='optional folder to pre-load (uploading still works)')
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()

    if args.image_dir:
        d = Path(args.image_dir)
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts:
                img = read_image(str(p))
                if img is not None:
                    _add_image(p.name, img)
        print(f'pre-loaded {len(IMAGES)} images from {args.image_dir}')

    print(f'cadrop web UI  ->  http://{args.host}:{args.port}')
    print('upload images in the browser, or pass --image-dir to pre-load')
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
