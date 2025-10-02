import cv2
import time


def _load_cascades():
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    smile_cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade: " + face_cascade_path)
    if smile_cascade.empty():
        raise RuntimeError("Failed to load smile cascade: " + smile_cascade_path)
    return face_cascade, smile_cascade


FACE_CASCADE, SMILE_CASCADE = _load_cascades()


def detect_smile_cascade(img, draw=True):
    """Detect smiles using OpenCV Haar cascades.

    Returns (img, is_smiling, num_faces, num_smiles)
    """
    if img is None or img.size == 0:
        return img, False, 0, 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(80, 80),
    )

    total_smiles = 0
    is_smiling_global = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        # Smile tends to need stronger filtering to reduce false positives
        smiles = SMILE_CASCADE.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=22,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(25, 25),
        )

        total_smiles += len(smiles)
        if len(smiles) > 0:
            is_smiling_global = True

        if draw:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(img, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 200, 255), 2)

    if draw:
        text = "Smiling" if is_smiling_global else "Not smiling"
        cv2.putText(img, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0) if is_smiling_global else (0, 0, 200), 2)

    return img, is_smiling_global, len(faces), total_smiles


def _run_webcam_demo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0.0
    last_state = False
    cooldown_seconds = 0.7
    last_emit = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame, is_smiling, num_faces, num_smiles = detect_smile_cascade(frame, draw=True)

        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Debounced printing to console
        if is_smiling != last_state and (now - last_emit) > cooldown_seconds:
            last_emit = now
            last_state = is_smiling
            if is_smiling:
                print("😀 Smile detected (cascade)!")
            else:
                print("🙂 Not smiling (cascade)")

        cv2.imshow("Haar Cascade Smile Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_webcam_demo()


