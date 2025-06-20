import cv2
import os
from datetime import datetime

# === CONFIG =======
SAVE_DIR = "data"
LABEL = "walking"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
# ==================

# Checks that the lable directory exists
save_path = os.path.join(SAVE_DIR, LABEL)
os.makedirs(save_path, exist_ok = True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open the webcam...")
    exit()

print(f"[I] Collecting images for label: '{LABEL}'.")
print(f"[I] Press 's' to save and image, and 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame...")
        break

    frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    cv2.imshow("Cat Cam", frame_resized)

    action = cv2.waitKey(1)
    if action == ord('s'):
        # Generate a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_path, f"{timestamp}.jpg")
        cv2.imwrite(filename, frame_resized)
        print(f"Saved pic as ... {filename}")

    elif action == ord('q'):
        print("[I] Quitting...")
        break

cap.release()
cv2.destroyAllWindows()


