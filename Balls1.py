import cv2
import numpy as np
from math import pi
from pathlib import Path

IMAGE_PATH = "C:/Users/nuamy/Downloads/Telegram Desktop/hm4.jpg"
SAVE_PATH  = "detected_output3.png"

SAT_MIN, VAL_MIN  = 25, 35
SAT_BOOST         = 1.15
CLAHE_CLIP        = 2.0

RADIUS_MIN, RADIUS_MAX = 25, 120

CIRC_MIN_STRICT = 0.55
CIRC_MIN_SOFT   = 0.52
ASPECT_TOL      = 0.35
SOLIDITY_MIN    = 0.65
EXTENT_MIN      = 0.55

COLORFULNESS_MIN = 10.0
HUE_ENTROPY_MIN  = 0.45
EXCLUDE_GREEN    = True
GREEN_H_LO, GREEN_H_HI = 40, 80

def colorfulness_metric(bgr):
    B, G, R = cv2.split(bgr.astype(np.float32))
    rg = np.abs(R - G); yb = np.abs(0.5*(R + G) - B)
    return np.hypot(rg.std(), yb.std()) + 0.3*np.hypot(rg.mean(), yb.mean())

def hue_entropy(hsv, mask=None, bins=24):
    H = hsv[..., 0]
    if mask is None: mask = np.ones(H.shape, np.uint8)*255
    hist = cv2.calcHist([H], [0], mask, [bins], [0, 180]).ravel()
    p = hist / (hist.sum() + 1e-9); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def is_greenish(hsv, mask=None, lo=40, hi=80):
    H = hsv[...,0]
    if mask is None: mask = np.ones(H.shape, np.uint8)*255
    h = H[mask > 0]
    return h.size > 0 and (np.mean((h >= lo) & (h <= hi)) > 0.55)

def preprocess_hsv(bgr):
    blur = cv2.GaussianBlur(bgr, (5,5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8,8))
    v = clahe.apply(v)
    s = np.clip(s.astype(np.float32)*SAT_BOOST, 0, 255).astype(np.uint8)
    return cv2.merge([h,s,v])

def detect_balls(image_bgr):
    hsv  = preprocess_hsv(image_bgr)
    mask = cv2.inRange(hsv, (0, SAT_MIN, VAL_MIN), (180, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = image_bgr.copy()
    kept = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50: 
            continue
        (cx,cy), r = cv2.minEnclosingCircle(c); r = float(r)
        if r < RADIUS_MIN or r > RADIUS_MAX:
            continue

        perim = cv2.arcLength(c, True) + 1e-6
        circularity = 4.0 * np.pi * area / (perim * perim)
        x,y,w,h = cv2.boundingRect(c)
        aspect   = min(w,h) / (max(w,h) + 1e-6)
        extent   = area / (w*h + 1e-6)
        hull     = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-6)

        good_shape = (circularity >= CIRC_MIN_STRICT) or (
            circularity >= CIRC_MIN_SOFT and
            aspect   >= (1.0 - ASPECT_TOL) and
            solidity >= SOLIDITY_MIN and
            extent   >= EXTENT_MIN
        )
        if not good_shape:
            continue

        pad = int(r * 1.1)
        x0,y0 = max(int(cx)-pad,0), max(int(cy)-pad,0)
        x1,y1 = min(int(cx)+pad,vis.shape[1]-1), min(int(cy)+pad,vis.shape[0]-1)
        crop_bgr = vis[y0:y1+1, x0:x1+1]
        crop_hsv = hsv[y0:y1+1, x0:x1+1]
        yy,xx = np.ogrid[:(y1-y0+1), :(x1-x0+1)]
        circ_mask = (((xx-(cx-x0))**2 + (yy-(cy-y0))**2) <= (0.9*r)**2).astype(np.uint8)*255

        C    = colorfulness_metric(crop_bgr)
        Hent = hue_entropy(crop_hsv, circ_mask)
        if C < COLORFULNESS_MIN or Hent < HUE_ENTROPY_MIN:
            continue
        if EXCLUDE_GREEN and is_greenish(crop_hsv, circ_mask, GREEN_H_LO, GREEN_H_HI):
            continue

        kept.append((int(cx), int(cy), int(r)))

    for (x,y,r) in kept:
        cv2.circle(vis, (x,y), r, (0,255,0), 2)
        cv2.circle(vis, (x,y), 2, (0,255,0), -1)

    return vis, mask, len(kept)

if __name__ == "__main__":
    if not Path(IMAGE_PATH).exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    img = cv2.imread(IMAGE_PATH)
    vis, mask, count = detect_balls(img)

    cv2.putText(vis, f"Detections: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"Detections: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    canvas   = np.ones((max(vis.shape[0], mask_bgr.shape[0]),
                        vis.shape[1] + mask_bgr.shape[1], 3), dtype=np.uint8)*255
    canvas[:vis.shape[0], :vis.shape[1]] = vis
    canvas[:mask_bgr.shape[0], vis.shape[1]:] = mask_bgr
    cv2.imwrite(SAVE_PATH, canvas)

    cv2.imshow("Detected Balls", vis)
    cv2.imshow("Color Mask", mask)
    print(f"Detected balls: {count}")
    print(f"Saved: {SAVE_PATH}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
