from __future__ import annotations

import sys
import cv2

try:
    # fer: Facial Emotion Recognition
    from fer import FER
except Exception as import_error:  # pragma: no cover
    print("Error: 'fer' package is not installed or failed to import.")
    print("Install it with: pip install fer")
    sys.exit(1)


def create_detector() -> FER:
    """
    Create a FER detector. Prefer MTCNN for better face detection; fallback to default
    if MTCNN dependencies are unavailable.
    """
    try:
        return FER(mtcnn=True)
    except Exception:
        # Fallback without MTCNN
        print("MTCNN unavailable, falling back to default FER detector.")
        return FER()


def main() -> None:
    detector = create_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (index 0).")
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Try to continue if a frame read fails
                continue

            # Detect emotions on the current frame
            try:
                result = detector.detect_emotions(frame)
            except Exception:
                # In rare cases, detection can throw on malformed frames; skip frame
                result = []

            if result:
                emotions = result[0].get("emotions", {})
                # Confidence threshold for happiness/smile
                if emotions.get("happy", 0.0) > 0.6:
                    cv2.putText(
                        frame,
                        "SMILE :)",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                        lineType=cv2.LINE_AA,
                    )
                    print("\U0001F600 Smile detected!")

            cv2.imshow("img", frame)
            key = cv2.waitKey(1) & 0xFF
            # ESC (27) or 'q' to quit
            if key in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


