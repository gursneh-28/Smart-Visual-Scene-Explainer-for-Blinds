# Real-world heights (cm) of common COCO objects — used for distance estimation
KNOWN_HEIGHTS_CM = {
    'person':        170,
    'man':           170,
    'woman':         170,
    'child':          90,
    'bottle':         25,
    'wine glass':     20,
    'cup':            12,
    'bowl':           10,
    'cell phone':     15,
    'remote':         18,
    'keyboard':        4,
    'mouse':           5,
    'laptop':         25,
    'tv':             60,
    'monitor':        40,
    'book':           25,
    'clock':          30,
    'vase':           30,
    'backpack':       50,
    'handbag':        30,
    'umbrella':      100,
    'chair':          85,
    'couch':          85,
    'bed':            60,
    'dining table':   75,
    'desk':           75,
    'toilet':         40,
    'sink':           20,
    'refrigerator':  180,
    'oven':           90,
    'microwave':      35,
    'toaster':        20,
    'car':           150,
    'truck':         250,
    'bus':           300,
    'bicycle':       100,
    'motorcycle':    110,
    'traffic light':  80,
    'fire hydrant':   60,
    'stop sign':      75,
    'dog':            50,
    'cat':            30,
    'bird':           20,
    'horse':         150,
    'cow':           140,
    'elephant':      280,
    'bear':          110,
    'zebra':         140,
    'giraffe':       500,
    'banana':         20,
    'apple':          10,
    'orange':         10,
    'carrot':         20,
    'hot dog':        12,
    'pizza':           5,
    'sandwich':       10,
    'cake':           15,
    'scissors':       20,
    'knife':          25,
    'fork':           20,
    'spoon':          18,
    'tie':            50,
    'suitcase':       65,
    'sports ball':    22,
    'kite':           60,
}
DEFAULT_HEIGHT_CM = 40  # Fallback for unknown objects

# Approximate focal length in pixels for a typical smartphone at 720p
# Using ~70° vertical FOV: focal_px = (720/2) / tan(35°) ≈ 514
FOCAL_LENGTH_PX = 514


def estimate_distance_meters(label, box, frame_height):
    """
    Physics-based distance estimate using pinhole camera model:
        distance = (real_height_cm * focal_px) / (pixel_height * 100)
    Returns distance in metres, rounded to 1 decimal place.
    """
    x1, y1, x2, y2 = box
    obj_height_px = max(y2 - y1, 1)

    real_height_cm = KNOWN_HEIGHTS_CM.get(label.lower(), DEFAULT_HEIGHT_CM)
    distance_m = (real_height_cm * FOCAL_LENGTH_PX) / (obj_height_px * 100)

    # Cap to a sensible range
    distance_m = max(0.3, min(distance_m, 30.0))
    return round(distance_m, 1)


def distance_to_words(dist_m):
    """Convert numeric distance to a natural phrase."""
    if dist_m <= 0.8:
        return f"about {dist_m} metre away — very close"
    elif dist_m <= 2.0:
        return f"about {dist_m} metres away"
    elif dist_m <= 5.0:
        return f"about {dist_m} metres away — a few steps"
    elif dist_m <= 10.0:
        return f"about {dist_m} metres away"
    else:
        return f"about {dist_m} metres away — far"


def get_position(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Horizontal thirds
    if center_x < frame_width * 0.33:
        horizontal = "to your left"
    elif center_x > frame_width * 0.66:
        horizontal = "to your right"
    else:
        horizontal = "ahead of you"

    # Vertical thirds (useful for floor objects vs overhead)
    if center_y < frame_height * 0.33:
        vertical = "upper area"
    elif center_y > frame_height * 0.66:
        vertical = "lower area"
    else:
        vertical = ""

    return horizontal, vertical


def describe_scene(detected, frame_width, frame_height):
    if not detected:
        return []

    descriptions = []
    for obj in detected:
        horizontal, vertical = get_position(obj['box'], frame_width, frame_height)
        dist_m   = estimate_distance_meters(obj['label'], obj['box'], frame_height)
        dist_str = distance_to_words(dist_m)

        display  = obj.get('display_name', obj['label'])
        desc     = f"{display.capitalize()} {horizontal}, {dist_str}"

        descriptions.append({
            'label':       obj['label'],
            'display_name': display,
            'description': desc,
            'horizontal':  horizontal,
            'vertical':    vertical,
            'distance_m':  dist_m,
            'distance_str': dist_str,
            'confidence':  obj['confidence']
        })

    return descriptions