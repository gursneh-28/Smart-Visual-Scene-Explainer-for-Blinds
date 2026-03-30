from flask import Flask, render_template, request, jsonify
from detector import detect_objects

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'description': 'No image received.'})

    image_file = request.files['image']
    image_bytes = image_file.read()

    # Detect objects
    detected, frame = detect_objects(image_bytes)

    if not detected:
        description = "No objects detected in the scene."
    else:
        # Build description
        items = [f"{d['label']} ({d['confidence']}%)" for d in detected]
        unique_items = list(dict.fromkeys([d['label'] for d in detected]))
        description = "I can see: " + ", ".join(unique_items) + "."

    return jsonify({'description': description})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)