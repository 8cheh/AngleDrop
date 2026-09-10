"""Compare super-resolution (3x) against the original on a drop image.

Prints the measured contact angle for each variant and saves the upscaled
images next to the script so the improvement is visible by eye.

Usage:
    python compare_sr.py [path/to/image.jpg]
"""
import os
import sys
import time

import cv2

from cadrop import measure, sr_available, super_resolve


def report(tag, bgr):
    t0 = time.time()
    r = measure(bgr, filename=tag)
    dt = time.time() - t0
    if r.ok:
        print(f"[{tag:10s}] L={r.left.angle_deg:6.2f}  R={r.right.angle_deg:6.2f}  "
              f"avg={r.avg_angle:6.2f}  conf={r.confidence:.2f}  "
              f"size={r.image_size[0]}x{r.image_size[1]}  ({dt:.1f}s)")
    else:
        print(f"[{tag:10s}] FAILED: {r.error}  ({dt:.1f}s)")
    return r


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        '..', 'Summer vacation ver 4.6 (available)', 'example picture.jpg')
    if not os.path.isfile(path):
        print('image not found:', path)
        sys.exit(1)

    print('super-resolve available:', sr_available())
    bgr = cv2.imread(path)
    if bgr is None:
        print('could not read image:', path)
        sys.exit(1)
    print('input size:', bgr.shape[1], 'x', bgr.shape[0])
    print('-' * 70)

    report('original', bgr)

    if not sr_available():
        print('onnxruntime/model unavailable -- skipping super-resolution')
        sys.exit(0)

    for scale in (3,):
        t0 = time.time()
        up = super_resolve(bgr, scale=scale)
        dt = time.time() - t0
        out = f'sr_{int(scale)}x.jpg'
        cv2.imwrite(out, up)
        print(f"sr {scale}x -> {up.shape[1]}x{up.shape[0]}  "
              f"({dt:.1f}s), saved '{out}'")
        report(f'sr{scale}x', up)
        # verify the integrated use_sr flag matches the direct call
        r = measure(bgr, filename='flag', use_sr=True, sr_scale=scale)
        if r.ok:
            print(f"[sr{scale}x(flag)] avg={r.avg_angle:.2f} conf={r.confidence:.2f}")
        else:
            print(f"[sr{scale}x(flag)] FAILED: {r.error}")


if __name__ == '__main__':
    main()
