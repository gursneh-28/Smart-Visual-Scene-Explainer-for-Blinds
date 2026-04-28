from ultralytics import YOLO
import cv2
import numpy as np

# Load model once at startup — use nano for speed, or switch to yolov8s.pt for accuracy
model = YOLO('yolov8n.pt')

# Warm up the model so first inference isn't slow
_dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model(_dummy, verbose=False)
del _dummy

# ── Alias map: user speech → YOLO class names ──
FIND_ALIASES = {
    'mobile':         'cell phone',
    'phone':          'cell phone',
    'smartphone':     'cell phone',
    'iphone':         'cell phone',
    'android':        'cell phone',
    'cellphone':      'cell phone',
    'cell phone':     'cell phone',
    'sofa':           'couch',
    'couch':          'couch',
    'television':     'tv',
    'telly':          'tv',
    'screen':         'tv',
    'monitor':        'tv',
    'spectacles':     'glasses',
    'specs':          'glasses',
    'eyeglasses':     'glasses',
    'mug':            'cup',
    'auto':           'car',
    'bike':           'bicycle',
    'cycle':          'bicycle',
    'motorbike':      'motorcycle',
    'water bottle':   'bottle',
    'plastic bottle': 'bottle',
    'soda':           'bottle',
    'jar':            'bottle',
    'laptop':         'laptop',
    'notebook':       'laptop',
    'book':           'book',
    'handbag':        'handbag',
    'bag':            'handbag',
    'backpack':       'backpack',
    'rucksack':       'backpack',
    'wallet':         'handbag',
    'keys':           'cell phone',   # YOLO doesn't detect keys, best effort
    'glasses':        'glasses',
    'sunglasses':     'glasses',
    'plate':          'bowl',
    'food':           'sandwich',
    'drink':          'cup',
    'coffee':         'cup',
    'tea':            'cup',
    'remote':         'remote',
    'remote control': 'remote',
}

DISPLAY_NAMES = {
    'cell phone': 'mobile phone',
    'tv':         'television',
}

# Classes that are commonly small/far away — lower conf threshold for these
HARD_TO_DETECT = {'bottle', 'cell phone', 'remote', 'glasses', 'cup', 'book', 'keys'}

def normalize_find_target(target: str) -> str:
    """Normalize user's spoken target to a YOLO class name."""
    t = target.lower().strip()
    for prefix in ('the ', 'a ', 'an ', 'my ', 'some ', 'any '):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    return FIND_ALIASES.get(t, t)

def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)

def detect_objects(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_height, frame_width = frame.shape[:2]

    # ── Resize to 640px wide for fast YOLO inference ──
    scale = 640 / frame_width if frame_width > 640 else 1.0
    if scale < 1.0:
        small = cv2.resize(frame, (640, int(frame_height * scale)))
    else:
        small = frame

    # Run with lower confidence to catch small objects; NMS 0.45 prevents duplicates
    results = model(small, conf=0.25, iou=0.45, verbose=False)

    detected    = []
    label_count = {}

    for result in results:
        for box in result.boxes:
            label      = result.names[int(box.cls)]
            confidence = float(box.conf)

            # Extra filter: for common objects require standard confidence,
            # but for hard-to-detect objects allow lower threshold
            min_conf = 0.20 if label in HARD_TO_DETECT else 0.30
            if confidence < min_conf:
                continue

            # Scale box coords back to original frame size
            coords = box.xyxy[0].tolist()
            if scale < 1.0:
                coords = [c / scale for c in coords]

            label_count[label] = label_count.get(label, 0) + 1

            detected.append({
                'label':        label,
                'display_name': display_name(label),
                'confidence':   round(confidence * 100, 1),
                'box':          coords,
            })

    detected.sort(key=lambda x: x['confidence'], reverse=True)
    return detected, frame, frame_width, frame_height