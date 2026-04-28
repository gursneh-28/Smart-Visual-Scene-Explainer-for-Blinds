"""
spatial.py — Converts YOLO bounding boxes into human-readable spatial descriptions.

Distance estimation uses the apparent size of an object relative to its known real-world size.
This is a monocular depth estimation heuristic — not perfect but useful.
"""

# Known approximate real-world heights (metres) for common YOLO classes
KNOWN_HEIGHTS = {
    'person':     1.70,
    'car':        1.50,
    'truck':      3.00,
    'bus':        3.20,
    'motorcycle': 1.10,
    'bicycle':    1.00,
    'dog':        0.50,
    'cat':        0.30,
    'chair':      0.90,
    'couch':      0.85,
    'bed':        0.55,
    'dining table': 0.75,
    'desk':       0.75,
    'bottle':     0.25,
    'cup':        0.12,
    'cell phone': 0.14,
    'laptop':     0.25,
    'tv':         0.70,
    'remote':     0.16,
    'keyboard':   0.04,
    'book':       0.22,
    'backpack':   0.50,
    'handbag':    0.30,
    'suitcase':   0.65,
    'umbrella':   0.90,
    'clock':      0.30,
    'vase':       0.30,
    'plant':      0.60,
    'traffic light': 0.80,
    'stop sign':  0.75,
    'fire hydrant': 0.45,
}

DEFAULT_HEIGHT = 0.40  # fallback for unknown objects

# Camera vertical FOV (degrees) — typical phone camera
VERTICAL_FOV_DEG = 60.0

def _estimate_distance(label: str, box_height_px: float, frame_height_px: int) -> float:
    """
    Monocular distance estimate using pinhole camera model.
    distance = (real_height * focal_length) / apparent_pixel_height
    focal_length in pixels ≈ frame_height / (2 * tan(FOV/2))
    """
    import math
    real_h = KNOWN_HEIGHTS.get(label, DEFAULT_HEIGHT)
    fov_rad = math.radians(VERTICAL_FOV_DEG / 2)
    focal_px = frame_height_px / (2 * math.tan(fov_rad))

    if box_height_px < 1:
        return 99.0  # Avoid division by zero

    distance = (real_h * focal_px) / box_height_px
    return round(max(0.3, min(distance, 50.0)), 1)  # Clamp 0.3m – 50m

def _horizontal_position(box: list, frame_width: int) -> tuple[str, str]:
    """Returns (brief_label, detailed_label) for horizontal position."""
    cx = (box[0] + box[2]) / 2
    ratio = cx / frame_width

    if ratio < 0.25:
        return "far left", "on your far left"
    elif ratio < 0.4:
        return "left", "to your left"
    elif ratio < 0.6:
        return "centre", "directly ahead"
    elif ratio < 0.75:
        return "right", "to your right"
    else:
        return "far right", "on your far right"

def _vertical_position(box: list, frame_height: int) -> str:
    """Returns rough vertical zone."""
    cy = (box[1] + box[3]) / 2
    ratio = cy / frame_height
    if ratio < 0.33:
        return "high"
    elif ratio < 0.67:
        return "mid"
    else:
        return "low"

def _distance_label(distance_m: float) -> str:
    """Convert distance to friendly spoken label."""
    if distance_m < 0.8:
        return "very close — within arm's reach"
    elif distance_m < 2.0:
        return f"about {distance_m:.1f} metres away — nearby"
    elif distance_m < 5.0:
        return f"about {distance_m:.1f} metres away"
    elif distance_m < 15.0:
        return f"around {int(distance_m)} metres away"
    else:
        return "far away"

def describe_scene(detected: list, frame_width: int, frame_height: int) -> list[dict]:
    """
    Convert detected objects into rich spatial descriptions.
    Returns list of dicts with 'description', 'horizontal', 'distance_m', etc.
    """
    if not detected or frame_width == 0 or frame_height == 0:
        return []

    results = []
    seen_labels = {}  # label → count

    for obj in detected:
        label    = obj['label']
        box      = obj['box']  # [x1, y1, x2, y2]
        conf     = obj['confidence']

        box_h    = box[3] - box[1]
        box_w    = box[2] - box[0]
        distance = _estimate_distance(label, box_h, frame_height)
        dist_str = _distance_label(distance)

        horiz_short, horiz_long = _horizontal_position(box, frame_width)
        vert                    = _vertical_position(box, frame_height)
        disp_name               = obj['display_name']

        # Count occurrences
        seen_labels[label] = seen_labels.get(label, 0) + 1
        count = seen_labels[label]

        # Build natural description
        if distance < 1.0:
            description = f"There's a {disp_name} very close {horiz_long}."
        elif count > 1:
            description = f"Another {disp_name} is {horiz_long}, {dist_str}."
        else:
            description = f"There's a {disp_name} {horiz_long}, {dist_str}."

        results.append({
            'label':        label,
            'display_name': disp_name,
            'description':  description,
            'horizontal':   horiz_short,
            'vertical':     vert,
            'distance_m':   distance,
            'distance_str': dist_str,
            'confidence':   conf,
            'box':          box,
        })

    # Sort: closer objects first (most relevant)
    results.sort(key=lambda x: x['distance_m'])
    return results