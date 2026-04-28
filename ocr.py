import numpy as np
import cv2

# ── Lazy init — don't load EasyOCR at import time (it's slow) ──
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='.ocr_models', verbose=False)
    return _reader

def _enhance_for_ocr(frame: np.ndarray) -> np.ndarray:
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def read_text(image_bytes: bytes) -> list[dict]:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Smaller = much faster OCR on CPU
    h, w = frame.shape[:2]
    target_w = 800
    if w > target_w:
        scale = target_w / w
        frame = cv2.resize(frame, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    enhanced = _enhance_for_ocr(frame)
    reader   = _get_reader()

    results = reader.readtext(
        enhanced,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        decoder='greedy',  # Much faster than beamsearch
    )

    seen_texts      = set()
    extracted_texts = []

    for (bbox, text, confidence) in results:
        text_clean = text.strip()
        text_lower = text_clean.lower()

        if len(text_clean) < 2 or confidence < 0.25:
            continue

        norm = ''.join(c for c in text_lower if c.isalnum() or c.isspace()).strip()
        if not norm or norm in seen_texts:
            continue
        if any(norm in seen for seen in seen_texts):
            continue

        seen_texts.add(norm)
        extracted_texts.append({
            'text':       text_clean,
            'confidence': round(confidence * 100, 1),
        })

    extracted_texts.sort(key=lambda x: x['confidence'], reverse=True)
    return extracted_texts[:8]