import easyocr
import numpy as np
import cv2

# Initialize OCR reader — English + Hindi; will use GPU for much faster processing
reader = easyocr.Reader(['en', 'hi'], gpu=True)


def preprocess_for_ocr(frame):
    """Enhance image contrast and sharpness before OCR for better text detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE: adaptive histogram equalisation — brings out low-contrast text
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gentle denoise to reduce noise before OCR
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)

    # Sharpen to make text edges crisper
    kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Back to BGR so EasyOCR gets a normal 3-channel image
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def read_text(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Try on both the original and the preprocessed image, merge results
    preprocessed = preprocess_for_ocr(frame)

    results_raw  = reader.readtext(frame, detail=1, paragraph=False)
    results_proc = reader.readtext(preprocessed, detail=1, paragraph=False)

    seen_texts     = set()
    extracted_texts = []

    for (bbox, text, confidence) in results_raw + results_proc:
        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Skip very short or duplicate text
        if len(text_clean) < 2:
            continue
        if text_lower in seen_texts:
            continue
        # Lower threshold to 0.30 to catch more text
        if confidence < 0.30:
            continue

        seen_texts.add(text_lower)
        extracted_texts.append({
            'text':       text_clean,
            'confidence': round(confidence * 100, 1)
        })

    # Sort by confidence
    extracted_texts.sort(key=lambda x: x['confidence'], reverse=True)
    return extracted_texts