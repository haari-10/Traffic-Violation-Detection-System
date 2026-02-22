import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Load models
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=False)

# Video path
cap = cv2.VideoCapture(r"C:\Users\murth\Downloads\Cars_Moving_On_Road_Stock_Footage_-_Free_Download_720p.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'output_result.mp4',
    fourcc,
    20.0,
    (int(cap.get(3)), int(cap.get(4)))
)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = 1          # FAST playback
frame_time = 1 / fps

vehicle_classes = ["car", "bus", "truck", "motorcycle"]
prev_positions = {}

SPEED_LIMIT = 60

# Frame skipping (2x speed)
frame_skip = 2
frame_count = 0

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, int(height * 0.4)),
        (width, int(height * 0.4)),
        (width, height),
        (0, height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180,
        threshold=80, minLineLength=150, maxLineGap=80
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = detect_lanes(frame)
    h, w, _ = frame.shape

    lane1_end = w // 3
    lane2_end = 2 * w // 3

    cv2.line(frame, (lane1_end, 0), (lane1_end, h), (255, 255, 0), 2)
    cv2.line(frame, (lane2_end, 0), (lane2_end, h), (255, 255, 0), 2)

    results = model.track(frame, persist=True, verbose=False)

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            speed = 0
            if track_id in prev_positions:
                dy = cy - prev_positions[track_id]
                speed = abs(dy) / frame_time
            prev_positions[track_id] = cy

            speed_violation = speed > SPEED_LIMIT

            if cx < lane1_end:
                lane = 1
            elif cx < lane2_end:
                lane = 2
            else:
                lane = 3

            wrong_lane = (
                (class_name == "motorcycle" and lane == 3) or
                (class_name == "truck" and lane == 1)
            )

            plate_text = ""

            # OCR ONLY IF VIOLATION
            if (speed_violation or wrong_lane):
                plate_crop = frame[int(y2 - (y2 - y1) * 0.3):y2, x1:x2]
                if plate_crop.size > 0:
                    plate_crop = cv2.resize(plate_crop, None, fx=0.5, fy=0.5)
                    ocr_result = reader.readtext(plate_crop, detail=0)
                    if ocr_result:
                        plate_text = ocr_result[0]

            color = (0, 255, 0)
            label = f"{class_name} L{lane} Sp:{int(speed)}"

            if speed_violation and wrong_lane:
                color = (0, 0, 255)
                label += " | SPEED | WRONG LANE"
            elif speed_violation:
                color = (0, 0, 255)
                label += " | SPEED"
            elif wrong_lane:
                color = (0, 0, 255)
                label += " | WRONG LANE"

            if plate_text:
                label += f" | {plate_text}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    out.write(frame)
    cv2.imshow("Traffic Violation + Number Plate", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
