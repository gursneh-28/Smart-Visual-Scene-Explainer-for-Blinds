from flask import Flask, render_template, request, jsonify, Response
from detector import detect_objects, normalize_find_target
from ocr import read_text
from spatial import describe_scene
from llm import generate_description, generate_description_stream
from dotenv import load_dotenv
import concurrent.futures
import traceback
import hashlib
import time
import json

load_dotenv()

app = Flask(__name__)

# ── Simple in-memory scene cache (avoids redundant Gemini calls) ──
scene_cache = {}
CACHE_TTL = 8  # seconds — if same scene hash within 8s, reuse description

def image_hash(image_bytes):
    """Fast perceptual hash to detect identical frames."""
    return hashlib.md5(image_bytes[::50]).hexdigest()  # Sample every 50th byte for speed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ping')
def ping():
    """Health check — client calls this on load to confirm server is up."""
    return jsonify({'status': 'ok'})

@app.route('/analyze', methods=['POST'])
def analyze():
    t_start = time.time()
    try:
        if 'image' not in request.files:
            return jsonify({'description': 'No image received.', 'objects': [], 'texts': []})

        image_file  = request.files['image']
        image_bytes = image_file.read()
        find_object = request.form.get('find', None)
        mode        = request.form.get('mode', 'full')  # full | fast | text_only

        if find_object and find_object != 'text_only':
            find_object = normalize_find_target(find_object)

        # ── Cache check (skip for targeted find requests) ──
        h = image_hash(image_bytes)
        if not find_object and h in scene_cache:
            cached = scene_cache[h]
            if time.time() - cached['ts'] < CACHE_TTL:
                return jsonify({**cached['data'], 'cached': True})

        # ── CRITICAL: Only run OCR when explicitly requested ──
        # EasyOCR on CPU takes 15-25s. Never run it for regular scene descriptions.
        run_ocr = (find_object == 'text_only')

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_detect = executor.submit(detect_objects, image_bytes)
            future_ocr    = executor.submit(read_text, image_bytes) if run_ocr else None

            detected, frame, frame_width, frame_height = future_detect.result(timeout=20)
            texts = future_ocr.result(timeout=55) if future_ocr else []

        spatial_descriptions = describe_scene(detected, frame_width, frame_height)

        # ── LLM Description ──
        description = generate_description(
            image_bytes, spatial_descriptions, texts, find_object
        )

        # ── Fallback if LLM fails ──
        if not description:
            description = _build_fallback(spatial_descriptions, texts, find_object)

        result = {
            'description': description,
            'objects':     spatial_descriptions[:8],
            'texts':       texts[:5],
            'latency_ms':  round((time.time() - t_start) * 1000)
        }

        # Cache the result
        if not find_object:
            scene_cache[h] = {'data': result, 'ts': time.time()}
            # Prune old cache entries
            if len(scene_cache) > 50:
                oldest = min(scene_cache, key=lambda k: scene_cache[k]['ts'])
                del scene_cache[oldest]

        return jsonify(result)

    except concurrent.futures.TimeoutError:
        return jsonify({
            'description': 'Analysis timed out — the scene might be too complex. Please try again.',
            'objects': [], 'texts': [], 'error': 'timeout'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'description': 'I had trouble analysing the scene. Please try again.',
            'objects': [], 'texts': [], 'error': str(e)
        })

def _build_fallback(spatial_descriptions, texts, find_object):
    parts = []
    if find_object and find_object != 'text_only':
        find_lower = find_object.lower()
        matches = [s for s in spatial_descriptions
                   if find_lower in s['label'].lower() or s['label'].lower() in find_lower]
        if matches:
            m = matches[0]
            parts.append(f"{m['display_name'].capitalize()} found {m['horizontal']}, approximately {m['distance_m']} metres away.")
        else:
            parts.append(f"I cannot find {find_object} in the current view. Try moving the camera around.")
    elif find_object == 'text_only':
        parts.append("I can read: " + ". ".join([t['text'] for t in texts]) if texts else "No readable text found.")
    else:
        for item in spatial_descriptions[:4]:
            parts.append(item['description'])
        if texts:
            parts.append("Visible text: " + ", ".join([f'"{t["text"]}"' for t in texts[:3]]))
        if not parts:
            parts.append("The scene appears empty or unclear. Try moving closer to objects.")
    return ". ".join(parts)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)