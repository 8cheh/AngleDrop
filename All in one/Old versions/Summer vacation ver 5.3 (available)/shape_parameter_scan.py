#!/usr/bin/env python3
"""Shape-parameter (P_s) scan: can a surface tension be measured from these images?

Answers one question and nothing else: **is there enough gravity deformation
in these drops for the shape to carry surface-tension information?**

This matters because contact angle and surface tension make opposite demands on
drop size.  A contact-angle experiment wants a small drop so gravity does not
flatten it; a surface-tension measurement needs the drop deformed *by* gravity,
because gamma lives in the deviation from sphericity.  A near-spherical drop is
therefore a perfectly good contact-angle image and a useless tensiometry image.

The diagnostic is the shape parameter, the fraction of the projected drop area
that lies outside the inscribed circle of radius equal to the apex radius of
curvature.  It is zero for a sphere and grows with deformation.  It is
scale-free and liquid-independent, so it can be computed from images that
already exist -- no calibration, no hardware, no Young-Laplace fit.

Published critical values (0.1 mJ/m^2 error target, Hoorfar):

    pendant drop, 3 mm capillary      0.29 (del Rio) / 0.35 (Rotenberg)
    pendant drop, 4 mm holder         0.19 / 0.22
    constrained sessile drop          0.18  (0.03 at a relaxed 1 mJ/m^2 target)

A free sessile drop at a low contact angle sits far below these.

Usage
-----
    python shape_parameter_scan.py --dir "D:\\path\\to\\images"
    python shape_parameter_scan.py --image drop.jpg --csv out.csv

Interpretation
--------------
    P_s >= 0.25   good -- a surface tension fit is worth attempting
    0.15 - 0.25   marginal -- try it, but check the fit residuals
    P_s <  0.15   the shape carries very little gamma information; expect a
                  large error, and do not report a surface tension
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

from cadrop.detect import read_image
from cadrop.geometry import Line
from cadrop.measure import measure
from cadrop.tensiometry import shape_parameter_from_profile

EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

#: thresholds used for the verdict column
GOOD = 0.25
MARGINAL = 0.15


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def list_images(directory):
    try:
        return [os.path.join(directory, f)
                for f in sorted(os.listdir(directory), key=natural_key)
                if os.path.splitext(f)[1].lower() in EXTS]
    except OSError:
        return []


def scan_one(path):
    """Return a result dict for one image, or one carrying an error."""
    row = {'file': os.path.basename(path), 'ok': False, 'error': '',
           'ps': None, 'r0_px': None, 'height_px': None, 'angle_deg': None}
    try:
        img = read_image(path)
        if img is None:
            row['error'] = 'unreadable'
            return row
        res = measure(img, os.path.basename(path), method='auto')
        if not res.ok or res.profile_xy is None:
            row['error'] = res.error or 'measurement failed'
            return row

        w = img.shape[1]
        line = Line(res.baseline_k, res.baseline_b, w, invert=False)
        rz = line.to_rz(res.profile_xy)

        ps, r0 = shape_parameter_from_profile(rz)
        if ps is None:
            row['error'] = 'profile unusable for P_s'
            return row

        row.update(ok=True, ps=float(ps), r0_px=float(r0),
                   height_px=float(rz[1].max()) if rz.size else None,
                   angle_deg=res.avg_angle)
    except Exception as exc:                      # keep the scan going
        row['error'] = f'{type(exc).__name__}: {exc}'
    return row


def verdict(ps):
    if ps is None:
        return 'n/a'
    if ps >= GOOD:
        return 'GOOD'
    return 'marginal' if ps >= MARGINAL else 'insufficient'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=None, help='folder of images')
    ap.add_argument('--image', default=None, help='single image')
    ap.add_argument('--csv', default=None, help='write results to this CSV')
    args = ap.parse_args()

    if args.image:
        paths = [args.image]
    elif args.dir:
        paths = list_images(args.dir)
        if not paths:
            print(f'no images found in {args.dir}')
            return 1
    else:
        print('give --dir or --image (see --help)')
        return 2

    print(f'scanning {len(paths)} image(s) ...\n')
    rows = [scan_one(p) for p in paths]

    print(f"  {'file':<34} {'P_s':>7} {'R0 px':>8} {'height':>7} "
          f"{'angle':>7}  verdict")
    print('  ' + '-' * 78)
    for r in rows:
        if r['ok']:
            print(f"  {r['file'][:34]:<34} {r['ps']:>7.3f} {r['r0_px']:>8.1f} "
                  f"{(r['height_px'] or 0):>7.0f} "
                  f"{(r['angle_deg'] if r['angle_deg'] is not None else float('nan')):>7.1f}  "
                  f"{verdict(r['ps'])}")
        else:
            print(f"  {r['file'][:34]:<34} {'-':>7} {'-':>8} {'-':>7} {'-':>7}  "
                  f"skipped: {r['error']}")

    good = [r for r in rows if r['ok'] and r['ps'] >= GOOD]
    marginal = [r for r in rows if r['ok'] and MARGINAL <= r['ps'] < GOOD]
    poor = [r for r in rows if r['ok'] and r['ps'] < MARGINAL]
    failed = [r for r in rows if not r['ok']]
    ok_ps = [r['ps'] for r in rows if r['ok']]

    print('\n' + '=' * 80)
    if not ok_ps:
        print('No image produced a usable profile -- nothing to conclude.')
        return 1

    print(f'  measured             : {len(ok_ps)}/{len(rows)}')
    print(f'  P_s median           : {np.median(ok_ps):.3f}')
    print(f'  P_s range            : {min(ok_ps):.3f} .. {max(ok_ps):.3f}')
    print(f'  good (>= {GOOD:.2f})      : {len(good)}')
    print(f'  marginal             : {len(marginal)}')
    print(f'  insufficient (< {MARGINAL:.2f}): {len(poor)}')
    print(f'  skipped              : {len(failed)}')
    print()

    med = float(np.median(ok_ps))
    if med >= GOOD:
        print('  VERDICT: the shapes carry real gravity deformation.')
        print('           A Young-Laplace fit has something to work with here.')
    elif med >= MARGINAL:
        print('  VERDICT: marginal. Expect a surface tension but treat its error')
        print('           bar as large; report it alongside the fit residual.')
    else:
        print('  VERDICT: these drops are too close to spherical.')
        print('           Contact angles from them are fine, but a surface tension')
        print('           would be dominated by the conditioning, not the camera.')
        print('           Options: larger drops, larger holder, or pendant geometry.')

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=['file', 'ok', 'ps', 'r0_px',
                                               'height_px', 'angle_deg',
                                               'verdict', 'error'])
            w.writeheader()
            for r in rows:
                d = dict(r)
                d['verdict'] = verdict(r['ps']) if r['ok'] else 'n/a'
                w.writerow({k: d.get(k) for k in w.fieldnames})
        print(f'\nwrote {args.csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
