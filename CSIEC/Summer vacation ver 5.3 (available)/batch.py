#!/usr/bin/env python3
"""Batch contact-angle measurement, CSV export and an annotated contact sheet.

Usage:
    python batch.py                          # measure ../Summer .../pictures -> batch_results.csv
    python batch.py --dir <folder> --out out.csv --method auto --sheet sheet.jpg
"""
import argparse
import csv
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from cadrop import METHODS, measure_file
from cadrop.detect import read_image


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def list_images(directory):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    try:
        return [os.path.join(directory, f) for f in sorted(os.listdir(directory), key=natural_key)
                if os.path.splitext(f)[1].lower() in exts]
    except OSError:
        return []


def _one(args):
    path, method, use_enhance = args
    res = measure_file(path, method=method, use_enhance=use_enhance)
    return res.to_dict()


def run(paths, method, use_enhance, workers=None):
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    tasks = [(p, method, use_enhance) for p in paths]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_one, tasks, chunksize=4))
    return [_one(t) for t in tasks]


def write_csv(results, out):
    fields = ['filename', 'ok', 'error', 'method',
              'left_angle', 'right_angle', 'avg_angle', 'asymmetry',
              'base_width_px', 'height_px', 'baseline_tilt_deg', 'confidence',
              'left_rms', 'right_rms', 'n_profile']
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            w.writerow([
                r['filename'], r['ok'], r.get('error', ''), r['method'],
                r['left'].get('angle_deg'), r['right'].get('angle_deg'),
                r.get('avg_angle'), r.get('asymmetry'),
                r.get('base_width_px'), r.get('height_px'),
                r['baseline']['tilt_deg'], r.get('confidence'),
                r['left'].get('rms_px'), r['right'].get('rms_px'),
                r.get('n_profile'),
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=None)
    ap.add_argument('--out', default='batch_results.csv')
    ap.add_argument('--method', default='auto', choices=METHODS)
    ap.add_argument('--enhance', action='store_true')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--sheet', default='contact_sheet.jpg', help='annotated grid (optional)')
    args = ap.parse_args()

    directory = args.dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'Summer Vacation ver,alpha 3.5 Data', 'pictures')
    paths = list_images(directory)
    if not paths:
        print(f'no images found in {directory}')
        return 1
    print(f'measuring {len(paths)} images (method={args.method}) ...')
    t0 = time.time()
    results = run(paths, args.method, args.enhance, args.workers)
    dt = time.time() - t0
    ok = sum(1 for r in results if r['ok'])
    print(f'done: {ok}/{len(results)} ok in {dt:.1f}s '
          f'({dt/len(results)*1000:.0f} ms/img)')
    write_csv(results, args.out)
    print(f'wrote {args.out}')

    if args.sheet:
        from cadrop.render import annotate
        cols = 5
        cells, cell_h = [], 380
        for p, r in zip(paths, results):
            img = read_image(p)
            if img is None or not r['ok']:
                continue
            res = measure_file(p, method=args.method, use_enhance=args.enhance)
            vis = annotate(img, res, show_panel=True)
            scale = cell_h / vis.shape[0]
            vis = cv2.resize(vis, (int(vis.shape[1] * scale), cell_h))
            cells.append(vis)
            if len(cells) == cols:
                break
        if cells:
            while len(cells) % cols:
                cells.append(np.zeros_like(cells[0]))
            rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
            sheet = np.vstack(rows)
            cv2.imwrite(args.sheet, sheet)
            print(f'wrote {args.sheet}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
