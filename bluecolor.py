import cv2
import numpy as np

# --- Параметрлер ---
IMAGE_PATH = "C:/Users/nuamy/Downloads/Telegram Desktop/hm4.jpg"
LOW_BLUE  = np.array([90, 70, 90])     # көк диапазон
HIGH_BLUE = np.array([130, 255, 255])
RADIUS_MIN, RADIUS_MAX = 15, 90        # доп өлшемі диапазоны

# --- Суретті оқу және HSV түрлендіру ---
img = cv2.imread(IMAGE_PATH)
hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5,5), 0), cv2.COLOR_BGR2HSV)

# --- Маска тек көк түске ---
mask = cv2.inRange(hsv, LOW_BLUE, HIGH_BLUE)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# --- Көк түстерді біріктіру және контур табу ---
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

vis = img.copy()
ball_count = 0
for i in range(1, num_labels):  # label 0 — фон
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 100:
        continue

    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[i]
    r = max(w, h) / 2

    # Егер радиус доптың өлшеміне сай болса — есептеу
    if RADIUS_MIN < r < RADIUS_MAX:
        cv2.circle(vis, (int(cx), int(cy)), int(r), (255, 0, 0), 2)
        ball_count += 1

print(f"Табылған көк топтар саны: {ball_count}")

cv2.imshow("Blue mask", mask)
cv2.imshow("Detected blue clusters", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
