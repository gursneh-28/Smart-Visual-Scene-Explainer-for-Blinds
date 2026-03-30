import cv2

def start_camera():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    print("✅ Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to grab frame.")
            break

        cv2.imshow("Visual Scene Explainer - Camera Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera stopped.")

if __name__ == "__main__":
    start_camera()