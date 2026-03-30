from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model (downloads automatically on first run)
model = YOLO('yolov8n.pt')  # 'n' = nano, fastest model

def detect_objects(image_bytes):
    # Convert bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run detection
    results = model(frame, conf=0.4)  # 0.4 = 40% confidence threshold

    detected = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            confidence = float(box.conf)
            detected.append({
                'label': label,
                'confidence': round(confidence * 100, 1),
                'box': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            })

    return detected, frame