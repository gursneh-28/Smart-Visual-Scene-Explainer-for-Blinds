# 👁️ Visual Scene Explainer for the Blind
An AI-powered assistive web app that uses real-time camera input to detect objects, read text, and describe surroundings via voice — accessible from any phone or browser.

> **Team Members:**
> - Gursneh Kaur (2023 BTech 033)
> - Kanishk Jain (2023 BTech 040)
> - Tanishka Nagal (2023 BTech 089)

---

## 🧰 Prerequisites

Before you begin, make sure you have the following installed:

- [Python 3.10+](https://www.python.org/downloads/)
- pip (comes with Python)
- A working webcam or mobile browser with camera access
- Git

---

## 🚀 Getting Started (After `git pull`)

### 1️⃣ Navigate to the project folder

---

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
```

**Activate it:**

- **Windows:**
  ```cmd
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

You should see `(venv)` at the start of your terminal. ✅

---

### 3️⃣ Install all dependencies
```bash
pip install ultralytics easyocr opencv-python pyttsx3 SpeechRecognition pyaudio google-generativeai pillow numpy flask flask-socketio
```
---

### 4️⃣ Verify Installation
```bash
python -c "import cv2; import easyocr; import pyttsx3; import speech_recognition; from ultralytics import YOLO; print('All good!')"
```
You should see **`All good!`** ✅

---

### 5️⃣ Run the App
```bash
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

---

## 🌐 Testing on Mobile (using ngrok)

1. Download ngrok from [https://ngrok.com/download](https://ngrok.com/download)

2. Sign up for a free account and get your auth token from [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)

3. Add your auth token:
   ```cmd
   ngrok.exe config add-authtoken YOUR_TOKEN_HERE
   ```

4. Run ngrok in a **separate terminal** (keep `app.py` running):
   ```cmd
   ngrok.exe http 5000
   ```

5. Copy the `https://...` URL shown and open it on your phone ✅

---