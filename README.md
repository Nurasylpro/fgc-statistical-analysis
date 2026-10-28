# FGC Statistical Analysis

Finds and counts colored balls in photos using classic computer vision (no neural network), then runs statistics on the detections.

## Context

| | |
|---|---|
| **Type** | University project |
| **Built** | October 2025 |
| **Area** | Computer vision · statistics |

## How it works

- Boosts saturation and applies CLAHE so colors stay visible in shadows.
- Builds color masks in HSV and filters contours by radius, circularity, aspect ratio, solidity and extent.
- Rejects false positives with a colorfulness score, hue entropy, and a green-background filter.
- Saves an annotated image (`detected_output*.png`) and prints statistics about the detections.

## Files

| File | Purpose |
|---|---|
| `Balls1.py` | Ball detector with strict/soft circularity thresholds |
| `bluecolor.py` | Single-color detection experiment |
| `statistical.py`, `statistics2.py` … `statistics5.py` | Successive versions of the detector + statistics |
| `detected_output*.png` | Example results |

## Tech

Python · OpenCV · NumPy

## Run it

```bash
pip install opencv-python numpy
python statistics5.py   # set IMAGE_PATH at the top of the file first
```

## What I learned

Tuning a classic CV pipeline by hand: color spaces, morphology, contour shape metrics, and iterating on false positives.
