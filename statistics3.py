import cv2
import numpy as np
from math import pi, sqrt
from pathlib import Path

# ---------- INPUT IMAGE ----------
IMAGE_PATH = "C:/Users/nuamy/Downloads/hm5.png"   # change to your filename, e.g. "Снимок экрана 2025-10-16 181445.png"

# ---------- PARAMETERS ----------
RADIUS_MIN = 8
RADIUS_MAX = 22
SAT_MIN = 55
VAL_MIN = 50
CIRCULARITY_MIN = 0.05
FILL_RATIO_MIN  = 0.01
COLORFULNESS_MIN = 14.0
HUE_ENTROPY_MIN  = 0.9
EXCLUDE_GREEN = True
GREEN_H_LO, GREEN_H_HI = (40, 80)
# --------------------------------

# ---------- Helper functions ----------
def colorfulness_metric(bgr):
    B, G, R = cv2.split(bgr.astype(np.float32))
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)
    return sqrt(rg.std()**2 + yb.std()**2) + 0.3*sqrt(rg.mean()**2 + yb.mean()**2)

def hue_entropy(hsv, mask=None, bins=24):
    H = hsv[..., 0]
    if mask is None: mask = np.ones(H.shape, np.uint8)*255
    hist = cv2.calcHist([H], [0], mask, [bins], [0, 180]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def is_greenish(hsv, mask=None, lo=40, hi=80):
    H = hsv[..., 0]
    if mask is None: mask = np.ones(H.shape, np.uint8)*255
    h = H[mask > 0]
    if h.size == 0: return False
    return np.mean((h >= lo) & (h <= hi)) > 0.55
# --------------------------------------

def detect_balls(image_bgr):
    vis = image_bgr.copy()
    blur = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Step 1: bright + saturated = likely colorful area
    color_mask = cv2.inRange(hsv, (0, SAT_MIN, VAL_MIN), (180, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 25:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        r = float(r)
        if r < RADIUS_MIN or r > RADIUS_MAX:
            continue

        perim = cv2.arcLength(c, True) + 1e-6
        circularity = 4.0 * pi * area / (perim * perim)
        fill_ratio  = area / (pi * r * r + 1e-6)
        if circularity < CIRCULARITY_MIN or fill_ratio < FILL_RATIO_MIN:
            continue

        pad = int(r * 1.1)
        x0, y0 = max(int(cx) - pad, 0), max(int(cy) - pad, 0)
        x1, y1 = min(int(cx) + pad, vis.shape[1]-1), min(int(cy) + pad, vis.shape[0]-1)
        crop_bgr = vis[y0:y1+1, x0:x1+1]
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        yy, xx = np.ogrid[:(y1-y0+1), :(x1-x0+1)]
        circ_mask = (((xx-(cx-x0))**2 + (yy-(cy-y0))**2) <= (0.9*r)**2).astype(np.uint8)*255

        C = colorfulness_metric(crop_bgr)
        Hent = hue_entropy(crop_hsv, circ_mask)
        if C < COLORFULNESS_MIN or Hent < HUE_ENTROPY_MIN:
            continue
        if EXCLUDE_GREEN and is_greenish(crop_hsv, circ_mask, GREEN_H_LO, GREEN_H_HI):
            continue

        kept.append((int(cx), int(cy), int(r)))

    # Draw results
    for (x, y, r) in kept:
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

    return vis, len(kept), color_mask

# ---------- Main ----------
if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError("Failed to read image. Check format or path.")

vis, count, mask = detect_balls(image)

cv2.imshow("Original", image)
cv2.imshow("Detected Balls", vis)
cv2.imshow("Color Mask (debug)", mask)

print(f"Detected {count} colorful balls.")
cv2.waitKey(0)
cv2.destroyAllWindows()
