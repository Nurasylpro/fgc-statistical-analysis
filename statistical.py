import cv2
import numpy as np
from collections import deque

# ------------ BASIC SETTINGS ------------
VIDEO_SOURCE = "C:/Users/nuamy/Downloads/human_player_1.mp4"   # change to video path if needed, e.g. "match.mp4"

# Ball size range in pixels (measure approximate ball radius in your frame!)
RADIUS_MIN, RADIUS_MAX = 8, 60

# Color filter (reduce glare / detect saturated colorful balls)
SAT_MIN, VAL_MIN = 30, 25

# Roundness geometry gates
CIRCULARITY_MIN = 0.35
FILL_RATIO_MIN  = 0.30

# Appearance gates (multi-color & vividness)
COLORFULNESS_MIN = 12.0
HUE_ENTROPY_MIN  = 0.30

# Exclusions
EXCLUDE_GREEN = True
GREEN_H_LO, GREEN_H_HI = 35, 85   # OpenCV hue 0..180

# ---------------------------------------

# --- ROI drawing (multiple rectangles) ---
rects, drawing, start_pt = [], False, None
def on_mouse(event, x, y, flags, ctx):
    global drawing, start_pt, rects
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing, start_pt = True, (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        ctx["preview"] = (start_pt, (x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0,y0,x1,y1 = start_pt[0],start_pt[1],x,y
        x0,x1 = sorted([x0,x1]); y0,y1 = sorted([y0,y1])
        if x1-x0>5 and y1-y0>5:
            rects.append((x0,y0,x1,y1))
        ctx["preview"] = None

def mask_from_rects(shape, rects):
    m = np.zeros(shape[:2], np.uint8)
    for (x0,y0,x1,y1) in rects: m[y0:y1, x0:x1] = 255
    return m
# -----------------------------------------

# ---- appearance / geometry helpers ----
def colorfulness_metric(bgr):
    B, G, R = cv2.split(bgr.astype(np.float32))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return np.sqrt(rg.std()**2 + yb.std()**2) + 0.3*np.sqrt(rg.mean()**2 + yb.mean()**2)

def hue_entropy(hsv, mask=None, bins=24):
    H = hsv[...,0]
    if mask is None: mask = np.ones(H.shape, np.uint8)*255
    hist = cv2.calcHist([H],[0],mask,[bins],[0,180]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p>0]
    return float(-(p*np.log2(p)).sum())

def is_greenish(hsv, mask=None):
    H = hsv[...,0]
    if mask is None: mask = np.ones(H.shape, np.uint8)*255
    h = H[mask>0]
    if h.size == 0: return False
    frac = np.mean((h>=GREEN_H_LO) & (h<=GREEN_H_HI))
    return frac > 0.55

recent = deque(maxlen=60)
def seen_recently(cx, cy, tol=12):
    for x,y in recent:
        if (x-cx)**2 + (y-cy)**2 <= tol*tol: return True
    recent.append((cx,cy)); return False
# -----------------------------------------

def detect_balls(frame, mask=None):
    vis = frame.copy()  # keep full frame (not cropped or dimmed)

    roi = frame if mask is None else cv2.bitwise_and(frame, frame, mask=mask)
    hsv = cv2.cvtColor(cv2.bilateralFilter(roi, 7, 60, 7), cv2.COLOR_BGR2HSV)

    # High saturation/value regions = potential balls
    color_mask = cv2.inRange(hsv, (0, SAT_MIN, VAL_MIN), (180, 255, 255))
    if mask is not None:
        color_mask = cv2.bitwise_and(color_mask, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 40: continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        r = float(r)
        if r < RADIUS_MIN or r > RADIUS_MAX: continue

        perim = cv2.arcLength(c, True) + 1e-6
        circularity = 4.0*np.pi*area/(perim*perim)
        fill_ratio  = area / (np.pi*r*r + 1e-6)
        if circularity < CIRCULARITY_MIN or fill_ratio < FILL_RATIO_MIN: continue

        pad = int(r*1.1)
        x0,y0 = max(int(cx)-pad,0), max(int(cy)-pad,0)
        x1,y1 = min(int(cx)+pad,frame.shape[1]-1), min(int(cy)+pad,frame.shape[0]-1)
        crop_bgr = frame[y0:y1+1, x0:x1+1]
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        yy,xx = np.ogrid[:(y1-y0+1),:(x1-x0+1)]
        circ_mask = (((xx-(cx-x0))**2 + (yy-(cy-y0))**2) <= (0.9*r)**2).astype(np.uint8)*255

        C = colorfulness_metric(crop_bgr)
        Hent = hue_entropy(crop_hsv, circ_mask)
        if C < COLORFULNESS_MIN or Hent < HUE_ENTROPY_MIN: continue
        if EXCLUDE_GREEN and is_greenish(crop_hsv, circ_mask): continue
        if seen_recently(int(cx), int(cy)): continue

        kept.append((int(cx), int(cy), int(r)))

    for (x,y,r) in kept:
        cv2.circle(vis,(x,y),r,(0,255,0),2)
        cv2.circle(vis,(x,y),2,(0,255,0),-1)

    return vis, len(kept), color_mask

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ok, frame = cap.read()
    if not ok: raise SystemExit("Cannot read first frame from source.")

    cv2.namedWindow("Draw ROIs: drag LMB; ENTER confirm; C clear; ESC full frame")
    ctx = {"preview": None}
    cv2.setMouseCallback("Draw ROIs: drag LMB; ENTER confirm; C clear; ESC full frame", on_mouse, ctx)

    while True:
        temp = frame.copy()
        for (x0,y0,x1,y1) in rects:
            cv2.rectangle(temp,(x0,y0),(x1,y1),(0,0,255),2)
        if ctx["preview"] is not None:
            p0,p1 = ctx["preview"]; cv2.rectangle(temp,p0,p1,(255,255,0),1)
        cv2.imshow("Draw ROIs: drag LMB; ENTER confirm; C clear; ESC full frame", temp)
        k = cv2.waitKey(10) & 0xFF
        if k == 13: break
        if k in (ord('c'), ord('C')): rects.clear()
        if k == 27: rects.clear(); break
    cv2.destroyAllWindows()

    roi_mask = mask_from_rects(frame.shape, rects) if rects else None

    while True:
        ok, frame = cap.read()
        if not ok: break
        vis, count, mask = detect_balls(frame, roi_mask)

        for (x0,y0,x1,y1) in rects:
            cv2.rectangle(vis,(x0,y0),(x1,y1),(255,255,255),2)

        cv2.putText(vis, f"Balls in ROIs: {count}", (18,36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Ball counter (final full frame)", vis)
        cv2.imshow("Mask (debug)", mask)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
