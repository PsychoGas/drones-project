import cv2
import time
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# Create a single long-lived FaceMesh instance (faster than recreating per frame)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def _to_pixel_coords(landmark, frame_width, frame_height):
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


def detect_smile(img, min_ratio: float = 1.0, max_ratio: float = 10.0, draw: bool = True):
    """Detect smile using mouth width/height ratio from MediaPipe Face Mesh.

    Returns (img, is_smiling, ratio) where ratio is mouth_width / mouth_height.
    """
    if img is None or img.size == 0:
        return img, False, 0.0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    is_smiling = False
    ratio_value = 0.0

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = img.shape

        # Mouth corner left (61), right (291), upper lip (13), lower lip (14)
        left = face_landmarks.landmark[61]
        right = face_landmarks.landmark[291]
        top = face_landmarks.landmark[13]
        bottom = face_landmarks.landmark[14]

        x_left, y_left = _to_pixel_coords(left, w, h)
        x_right, y_right = _to_pixel_coords(right, w, h)
        _, y_top = _to_pixel_coords(top, w, h)
        _, y_bottom = _to_pixel_coords(bottom, w, h)

        mouth_width = max(1, x_right - x_left)  # avoid zero division
        mouth_height = max(1, y_bottom - y_top)

        ratio_value = float(mouth_width) / float(mouth_height)
        is_smiling = (min_ratio <= ratio_value <= max_ratio)

        if draw:
            # Draw mouth key points and helper lines
            cv2.circle(img, (x_left, y_left), 2, (0, 255, 255), -1)
            cv2.circle(img, (x_right, y_right), 2, (0, 255, 255), -1)
            cv2.circle(img, (x_left, y_left), 6, (0, 255, 255), 1)
            cv2.circle(img, (x_right, y_right), 6, (0, 255, 255), 1)
            cv2.circle(img, (x_left, y_left), 10, (0, 255, 255), 1)
            cv2.circle(img, (x_right, y_right), 10, (0, 255, 255), 1)
            cv2.line(img, (x_left, y_left), (x_right, y_right), (0, 200, 255), 2)
            cv2.line(img, (x_left + (x_right - x_left) // 2, y_top), (x_left + (x_right - x_left) // 2, y_bottom), (0, 200, 255), 2)

            # Optional: draw the full mesh contours lightly
            mp_draw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
            )

            color = (0, 200, 0) if is_smiling else (0, 0, 200)
            cv2.putText(
                img,
                f"Smile ratio: {ratio_value:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

            if is_smiling:
                cv2.putText(
                    img,
                    "SMILE :)",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )

    return img, is_smiling, ratio_value


def _run_webcam_demo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0.0
    last_state = False
    cooldown_seconds = 0.6
    last_emit = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame, is_smiling, ratio_value = detect_smile(frame, min_ratio=1.0, max_ratio=10.0, draw=True)

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Emit on state change with cooldown
        if is_smiling != last_state and (now - last_emit) > cooldown_seconds:
            last_emit = now
            last_state = is_smiling
            if is_smiling:
                print("😀 Smile detected!")
            else:
                print("🙂 Not smiling")

        cv2.imshow("Smile Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_webcam_demo()


