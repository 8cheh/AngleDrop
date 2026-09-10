#!/usr/bin/env python3
"""
AngleAP — 本地接触角测量服务器
启动: python server.py
访问: http://localhost:8765

API:
  POST /api/opendrop  — OpenDrop 全自动接触角测量
  POST /api/polyfit   — v2.2 两点多项式拟合
"""

import os, json, base64, math, traceback, importlib.util
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=HERE)

# ---- 加载 batch_opendrop 模块 ----
spec = importlib.util.spec_from_file_location("batch_opendrop",
    os.path.join(HERE, "batch_opendrop.py"))
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


def img2b64(img_bgr):
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode() if ok else None


def read_img(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ---- OpenDrop 可视化 ----
def draw_od(vis, baseline, drop_pts, fit):
    h, w = vis.shape[:2]
    # 基线
    cv2.line(vis, (int(baseline.pt0.x), int(baseline.pt0.y)),
             (int(baseline.pt1.x), int(baseline.pt1.y)), (0, 0, 255), 2, cv2.LINE_AA)
    # 边缘点
    for x, y in drop_pts.astype(int).T:
        cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)
    # 接触点
    for key in ['left_contact', 'right_contact']:
        cp = fit.get(key)
        if cp is None: continue
        cx, cy = int(cp.x), int(cp.y)
        cv2.circle(vis, (cx, cy), 8, (255, 0, 0), -1)
        cv2.circle(vis, (cx, cy), 10, (255, 255, 255), 2)
        cv2.line(vis, (cx-12, cy), (cx+12, cy), (255, 0, 0), 2)
        cv2.line(vis, (cx, cy-12), (cx, cy+12), (255, 0, 0), 2)
    # 拟合弧线
    Q = np.array([baseline.unit, baseline.perp])
    pt0_arr = np.array([baseline.pt0.x, baseline.pt0.y])
    for ak, ck, cck in [('left_angle', 'left_curvature', 'left_arc_center'),
                          ('right_angle', 'right_curvature', 'right_arc_center')]:
        angle = fit.get(ak)
        curv = fit.get(ck)
        contact = fit.get(ak.replace('angle', 'contact'))
        center = fit.get(cck)
        if angle is None or contact is None: continue
        if curv and abs(curv) > 1e-9 and center:
            radius = 1.0 / abs(curv)
            c = np.array([center.x, center.y])
            crz = Q @ (c - pt0_arr)
            if abs(crz[1]) < radius:
                ha = math.acos(crz[1] / radius)
                crz2 = Q @ (np.array([contact.x, contact.y]) - pt0_arr)
                q0 = math.atan2(crz[1], crz[0] - crz2[0])
                sp = min(ha * 0.6, math.radians(30))
                sa, ea = (q0-sp, q0) if curv < 0 else (q0, q0+sp)
                angles = np.linspace(sa, ea, 30)
                pts = np.column_stack([c[0]+radius*np.cos(angles),
                                       c[1]+radius*np.sin(angles)]).astype(int)
                cv2.polylines(vis, [pts.reshape(-1,1,2)], False, (0,255,255), 2, cv2.LINE_AA)
    # 文字标注
    ld = math.degrees(fit.get('left_angle')) if fit.get('left_angle') else None
    rd = math.degrees(fit.get('right_angle')) if fit.get('right_angle') else None
    fs = max(0.7, min(1.2, w/1200))
    th = max(1, int(fs*2))
    yo = 30
    for txt, clr in [
        (f"L: {ld:.2f}" if ld else None, (255,200,100)),
        (f"R: {rd:.2f}" if rd else None, (100,255,200)),
        (f"Avg: {(ld+rd)/2:.2f}" if ld and rd else None, (255,255,100)),
    ]:
        if not txt: continue
        (tw, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs*0.85, th)
        ov = vis.copy()
        cv2.rectangle(ov, (10, yo), (14+tw, yo+th2+12), (0,0,0), -1)
        vis = cv2.addWeighted(vis, 0.6, ov, 0.4, 0)
        cv2.putText(vis, txt, (12, yo+th2), cv2.FONT_HERSHEY_SIMPLEX, fs*0.85, clr, th, cv2.LINE_AA)
        yo += th2 + 14
    return vis


# ---- v2.2 可视化 ----
def draw_poly(vis, p1, p2, le, re, lc, rc, la, ra):
    cv2.line(vis, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)
    for pts in [le, re]:
        if pts is not None and len(pts) > 0:
            for ex, ey in pts: cv2.circle(vis, (int(ex), int(ey)), 2, (0,255,0), -1)
    for pts, coeffs in [(le, lc), (re, rc)]:
        if pts is not None and len(pts) > 0 and coeffs:
            a, b, c = coeffs
            ys = np.linspace(np.min(pts[:,1]), np.max(pts[:,1]), 50)
            xs = (a*ys*ys + b*ys + c).astype(int)
            cv2.polylines(vis, [np.column_stack([xs, ys.astype(int)]).reshape(-1,1,2)],
                          False, (0,255,255), 2, cv2.LINE_AA)
    for pt, ang, lbl in [(p1, la, 'L'), (p2, ra, 'R')]:
        cv2.circle(vis, pt, 6, (255,0,0), -1)
        if ang is not None:
            cv2.putText(vis, f"{lbl}:{ang:.1f}",
                        (pt[0]-60 if lbl=='L' else pt[0]+10, pt[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
    return vis


# ============================================================
# API: OpenDrop 自动检测
# ============================================================
@app.route('/api/opendrop', methods=['POST'])
def api_opendrop():
    try:
        f = request.files.get('image')
        if not f: return jsonify({'error': '未上传图像'}), 400
        img = read_img(f.read())
        if img is None: return jsonify({'error': '无法解析图像'}), 400

        h, w = img.shape[:2]
        enhanced = bm.enhance_image(img)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        k, bv = bm.detect_baseline(gray, h, w)
        baseline = bm.Line2(bm.Vector2(0.0, k*0+bv),
                            bm.Vector2(float(w-1), k*(w-1)+bv))

        dp = bm.extract_contact_angle_features(enhanced, baseline, inverted=False, thresh=0.3)
        if dp.shape[1] < 10:
            vis = enhanced.copy()
            draw_od(vis, baseline, dp, {})
            return jsonify({
                'error': f'边缘点不足 ({dp.shape[1]}个)',
                'annotated_image': img2b64(vis),
                'left_angle': None, 'right_angle': None, 'avg_angle': None,
            })

        fit = bm.contact_angle_fit_opendrop(dp, baseline)
        la = round(math.degrees(fit['left_angle']), 2) if fit['left_angle'] else None
        ra = round(math.degrees(fit['right_angle']), 2) if fit['right_angle'] else None
        angles = [a for a in (la, ra) if a is not None]
        avg = round(sum(angles)/len(angles), 2) if angles else None

        vis = enhanced.copy()
        draw_od(vis, baseline, dp, fit)

        return jsonify({
            'method': 'OpenDrop (Python)',
            'left_angle': la, 'right_angle': ra, 'avg_angle': avg,
            'left_contact': {'x': int(fit['left_contact'].x), 'y': int(fit['left_contact'].y)} if fit.get('left_contact') else None,
            'right_contact': {'x': int(fit['right_contact'].x), 'y': int(fit['right_contact'].y)} if fit.get('right_contact') else None,
            'annotated_image': img2b64(vis),
            'baseline': {'k': float(k), 'b': float(bv)},
            'error': None if angles else '无法计算接触角',
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================
# API: v2.2 两点多项式拟合
# ============================================================
@app.route('/api/polyfit', methods=['POST'])
def api_polyfit():
    try:
        f = request.files.get('image')
        if not f: return jsonify({'error': '未上传图像'}), 400
        img = read_img(f.read())
        if img is None: return jsonify({'error': '无法解析图像'}), 400
        p1_x = int(float(request.form['p1_x']))
        p1_y = int(float(request.form['p1_y']))
        p2_x = int(float(request.form['p2_x']))
        p2_y = int(float(request.form['p2_y']))
    except (KeyError, ValueError):
        return jsonify({'error': '缺少接触点坐标'}), 400

    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)

        result = {}
        for pt, side, key in [((p1_x, p1_y), 'left', 'left'), ((p2_x, p2_y), 'right', 'right')]:
            cx, cy = pt
            ys, ye = max(0, cy-60), cy
            xs = max(0, cx-50) if side == 'left' else max(0, cx-20)
            xe = min(w-1, cx+20) if side == 'left' else min(w-1, cx+50)

            roi = blur[ys:ye, xs:xe]
            ep = []
            for yi, r in enumerate(roi):
                g = np.diff(r.astype(float))
                v = np.max(g) if side == 'left' else np.max(-g)
                if v > 15:
                    ep.append((int(xs + np.argmax(g if side == 'left' else -g)), int(ys + yi)))

            edge = np.array(ep) if ep else None
            result[key + '_edge'] = edge

            if edge is None or len(edge) < 5:
                result[key + '_angle'] = None
                result[key + '_coeffs'] = None
                continue

            try:
                a, b, c = np.polyfit(edge[:,1], edge[:,0], 2)
                yc = np.max(edge[:,1])
                tv = np.array([-2*a*yc - b, -1])
                bv_arr = np.array([1, 0]) if side == 'left' else np.array([-1, 0])
                angle = float(np.degrees(np.arccos(np.clip(
                    np.dot(tv, bv_arr) / (np.linalg.norm(tv) * np.linalg.norm(bv_arr)), -1, 1))))
                result[key + '_angle'] = angle
                result[key + '_coeffs'] = (float(a), float(b), float(c))
            except Exception:
                result[key + '_angle'] = None
                result[key + '_coeffs'] = None

        la = round(result['left_angle'], 2) if result.get('left_angle') else None
        ra = round(result['right_angle'], 2) if result.get('right_angle') else None
        angles = [a for a in (la, ra) if a is not None]
        avg = round(sum(angles)/len(angles), 2) if angles else None

        vis = img.copy()
        draw_poly(vis, (p1_x, p1_y), (p2_x, p2_y),
                  result.get('left_edge'), result.get('right_edge'),
                  result.get('left_coeffs'), result.get('right_coeffs'), la, ra)

        return jsonify({
            'method': 'v2.2 多项式拟合 (Python)',
            'left_angle': la, 'right_angle': ra, 'avg_angle': avg,
            'left_contact': {'x': p1_x, 'y': p1_y},
            'right_contact': {'x': p2_x, 'y': p2_y},
            'annotated_image': img2b64(vis),
            'error': None if angles else '无法计算接触角',
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================
# API: 健康检查
# ============================================================
@app.route('/api/health')
def api_health():
    return jsonify({'status': 'ok'})


# ============================================================
# 静态文件
# ============================================================
@app.route('/')
def index():
    return send_from_directory(HERE, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(HERE, path)


if __name__ == '__main__':
    print("=" * 50)
    print("  AngleAP Server")
    print("  http://localhost:8765")
    print("  API: /api/opendrop  |  /api/polyfit")
    print("=" * 50)
    app.run(host='127.0.0.1', port=8765, debug=False)
