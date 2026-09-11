# the best.pt file is generated using YOLO11 notebook in waste-classification folder 
# the original notebook is on kaggle 

import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION & METADATA
# ==========================================
MODEL_PATH = "best.pt"  # Path to your trained YOLO model
CAMERA_INDEX = 1        # 0 for PC webcam, 1 or 2 for Phone Link / DroidCam
CONF_THRESHOLD = 0.3   # Minimum detection threshold
STRICT_CONF = 0.40      # Below this confidence, target shows as 'ANALYZING...'

WASTE_INFO = {
    'plastic': {'decomp': '450 YRS', 'recycle': 'YES', 'hazard': 'LOW'},
    'glass': {'decomp': '1M YRS', 'recycle': 'YES', 'hazard': 'NONE'},
    'metal': {'decomp': '200 YRS', 'recycle': 'YES', 'hazard': 'MED'},
    'cardboard': {'decomp': '2 MOS', 'recycle': 'YES', 'hazard': 'NONE'},
    'paper': {'decomp': '6 WKS', 'recycle': 'YES', 'hazard': 'NONE'},
    'trash': {'decomp': 'VARIES', 'recycle': 'NO', 'hazard': 'HIGH'},
}

# ==========================================
# 2. HUD DRAWING HELPER FUNCTIONS
# ==========================================
def draw_terminator_box(img, x1, y1, x2, y2, color=(0, 0, 255), line_length=25, thickness=2):
    """Draws sci-fi corner brackets instead of standard rectangular bounding boxes."""
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)

# ==========================================
# 3. THREADING & CAMERA INITIALIZATION
# ==========================================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

latest_frame = None
boxes_cache = []
is_running = True

def inference_thread():
    """Background thread running YOLO prediction on full resolution without blocking video FPS."""
    global latest_frame, boxes_cache, is_running
    while is_running:
        if latest_frame is not None:
            img = latest_frame.copy()
            
            # Run inference at 640 resolution (Ultralytics auto-scales coordinates)
            results = model(img, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]

            new_boxes = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id].lower()
                new_boxes.append((x1, y1, x2, y2, class_name, conf))

            boxes_cache = new_boxes
            time.sleep(0.01)  # Throttle slightly to prevent high CPU utilization

# Launch background AI inference thread
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

# ==========================================
# 4. MAIN DISPLAY & RENDERING LOOP
# ==========================================
scanline_y = 0
prev_time = time.time()

print("[INFO] Terminator HUD active. Press 'q' to quit.")

while cap.isOpened():
    # Flush older buffered camera frames to eliminate transmission latency
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    latest_frame = frame
    h, w, _ = frame.shape

    # A. Red Animated Scanline Effect
    scanline_y = (scanline_y + 10) % h
    cv2.line(frame, (0, scanline_y), (w, scanline_y), (0, 0, 180), 1)

    # B. System Status & FPS Overlay (Top-Left)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, "[SYSTEM: TARGET ACQUISITION MODE]", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)} | TARGETS DETECTED: {len(boxes_cache)}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)

    # C. Render Object Overlay, Panels, and Target Crosshairs
    for x1, y1, x2, y2, class_name, conf in boxes_cache:
        # Handle Low Confidence or Out-of-Domain Objects
        if conf < STRICT_CONF:
            display_label = "ANALYZING..."
            decomp_val = "CALCULATING"
            recycle_val = "UNKNOWN"
        else:
            info = WASTE_INFO.get(class_name, {'decomp': 'N/A', 'recycle': 'UNKNOWN'})
            display_label = f"{class_name.upper()} ({conf*100:.0f}%)"
            decomp_val = info['decomp']
            recycle_val = info['recycle']

        # 1. Corner Brackets
        draw_terminator_box(frame, x1, y1, x2, y2, color=(0, 0, 255), line_length=25, thickness=2)

        # 2. Center Targeting Crosshair
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 1)
        cv2.line(frame, (cx - 8, cy), (cx + 8, cy), (0, 0, 255), 1)
        cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (0, 0, 255), 1)

        # 3. Position Metadata Panel (Keep text inside screen bounds)
        panel_w, panel_h = 210, 65
        text_x = x2 + 10 if x2 + panel_w < w else x1 - panel_w - 10
        text_y = max(y1, 10)

        # Dark Semi-Transparent Background Panel for Readability
        cv2.rectangle(frame, (text_x - 5, text_y), (text_x + panel_w, text_y + panel_h), (15, 15, 15), -1)
        cv2.rectangle(frame, (text_x - 5, text_y), (text_x + panel_w, text_y + panel_h), (0, 0, 255), 1)

        # Text Metadata Lines
        cv2.putText(frame, f">> {display_label}",
                    (text_x, text_y + 18), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"   DECOMP: {decomp_val}",
                    (text_x, text_y + 36), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"   RECYCLE: {recycle_val}",
                    (text_x, text_y + 52), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    # Display HUD Window
    cv2.imshow("TERMINATOR VISION - WASTE SCANNER", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        is_running = False
        break

cap.release()
cv2.destroyAllWindows()