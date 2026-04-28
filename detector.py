from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8l.pt')

# Maps user speech → YOLO class names
FIND_ALIASES = {
    'mobile':        'cell phone',
    'phone':         'cell phone',
    'smartphone':    'cell phone',
    'iphone':        'cell phone',
    'android':       'cell phone',
    'cellphone':     'cell phone',
    'cell phone':    'cell phone',
    'sofa':          'couch',
    'television':    'tv',
    'telly':         'tv',
    'screen':        'tv',
    'monitor':       'tv',
    'spectacles':    'glasses',
    'specs':         'glasses',
    'mug':           'cup',
    'auto':          'car',
    'bike':          'bicycle',
    'motorbike':     'motorcycle',
    'water bottle':  'bottle',
    'plastic bottle':'bottle',
}

# Friendly display names for YOLO labels
DISPLAY_NAMES = {
    'cell phone': 'mobile phone',
}

def normalize_find_target(target):
    """Normalize user's find request to YOLO class name."""
    t = target.lower().strip()
    return FIND_ALIASES.get(t, t)

def display_name(label):
    return DISPLAY_NAMES.get(label, label)


def detect_objects(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_height, frame_width = frame.shape[:2]

    # Lower conf to 0.35 so small/partially-visible objects (bottle, phone) get caught
    results = model(frame, conf=0.35, iou=0.5)

    detected   = []
    label_count = {}

    for result in results:
        for box in result.boxes:
            label      = result.names[int(box.cls)]
            confidence = float(box.conf)
            coords     = box.xyxy[0].tolist()

            # Count occurrences per label
            label_count[label] = label_count.get(label, 0) + 1

            detected.append({
                'label':        label,
                'display_name': display_name(label),
                'confidence':   round(confidence * 100, 1),
                'box':          coords
            })

    # Sort by confidence descending
    detected.sort(key=lambda x: x['confidence'], reverse=True)
    return detected, frame, frame_width, frame_height