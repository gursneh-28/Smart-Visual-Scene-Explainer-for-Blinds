from flask import Flask, render_template, request, jsonify
from detector import detect_objects, normalize_find_target
from ocr import read_text
from spatial import describe_scene
from llm import generate_description
from dotenv import load_dotenv
import concurrent.futures
import traceback

load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'description': 'No image received.', 'objects': [], 'texts': []})

        image_file  = request.files['image']
        image_bytes = image_file.read()
        find_object = request.form.get('find', None)

        # Normalize the find target on the server side too
        # so "mobile" → "cell phone" etc.
        if find_object and find_object != 'text_only':
            find_object = normalize_find_target(find_object)

        # Run YOLO and OCR in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_detect = executor.submit(detect_objects, image_bytes)
            future_ocr    = executor.submit(read_text, image_bytes)
            detected, frame, frame_width, frame_height = future_detect.result()
            texts = future_ocr.result()

        spatial_descriptions = describe_scene(detected, frame_width, frame_height)

        # Send actual image + context to Gemini Vision
        description = generate_description(image_bytes, spatial_descriptions, texts, find_object)

        # Fallback if Gemini fails
        if not description:
            parts = []
            if find_object and find_object != 'text_only':
                matches = [s for s in spatial_descriptions
                           if find_object.lower() in s['label'].lower()]
                if matches:
                    m = matches[0]
                    parts.append(
                        f"{m['display_name'].capitalize()} found {m['horizontal']}, "
                        f"approximately {m['distance_m']} metres away."
                    )
                else:
                    parts.append(f"Could not find {find_object} in view.")
            elif find_object == 'text_only':
                parts.append(
                    "I can read: " + ". ".join([t['text'] for t in texts])
                    if texts else "No text found in the scene."
                )
            else:
                for item in spatial_descriptions[:5]:
                    parts.append(item['description'])
                if texts:
                    parts.append("I can also read: " + ", ".join([t['text'] for t in texts[:5]]))
                if not parts:
                    parts.append("Nothing detected in the scene.")
            description = ". ".join(parts) + "."

        return jsonify({
            'description': description,
            'objects':     spatial_descriptions,
            'texts':       texts
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'description': 'Sorry, I had trouble analysing the scene. Please try again.',
            'objects':     [],
            'texts':       []
        })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)