#!/usr/bin/env python3
"""Fit a colour correction for the iMac camera from a reference photo.

Usage: calibrate.py CAMERA.jpg REFERENCE.jpg PATCHES.json
PATCHES.json: {"patches": [{"cam": [x0,y0,x1,y1], "ref": [x0,y0,x1,y1]}, ...]}
Each pair marks the same surface in both images (sunlit wall, sky, dome, brick, parapet...).
Fits RGB_ref = M * RGB_cam + offset by least squares and writes colour.json beside this script.
"""
import json
import sys

import cv2
import numpy as np

cam = cv2.imread(sys.argv[1])[:, :, ::-1].astype(np.float32)
ref = cv2.imread(sys.argv[2])[:, :, ::-1].astype(np.float32)
patches = json.load(open(sys.argv[3]))["patches"]
A, B = [], []
for p in patches:
    cx0, cy0, cx1, cy1 = p["cam"]
    rx0, ry0, rx1, ry1 = p["ref"]
    A.append(cam[cy0:cy1, cx0:cx1].reshape(-1, 3).mean(0))
    B.append(ref[ry0:ry1, rx0:rx1].reshape(-1, 3).mean(0))
A = np.array(A); B = np.array(B)
X = np.hstack([A, np.ones((len(A), 1))])          # [R G B 1]
coef, *_ = np.linalg.lstsq(X, B, rcond=None)      # 4x3
M = coef[:3].T; off = coef[3]
pred = X @ coef
for a, b, q in zip(A, B, pred):
    print("cam", a.round(), "ref", b.round(), "fit", q.round())
out = {"matrix": M.round(4).tolist(), "offset": off.round(2).tolist(), "patches": len(patches)}
json.dump(out, open(__file__.replace("calibrate.py", "colour.json"), "w"), indent=1)
print("wrote colour.json", out)
