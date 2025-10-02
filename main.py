import cv2
import time
import mediapipe as mp
import handtrackingmin as htm
import math

############################################
wCam, hCam = 640, 480
############################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector()

# Detection cooldown to avoid rapid re-triggering of gestures
DETECTION_COOLDOWN_SECONDS = 0.6
_last_detection_time = 0.0

def detect_and_emit(text: str) -> None:
    """Emit a command only if the detection cooldown has elapsed."""
    global _last_detection_time
    now = time.time()
    if (now - _last_detection_time) < DETECTION_COOLDOWN_SECONDS:
        return
    _last_detection_time = now
    print(text)

# Simple N-frame stability filter per gesture
STABILITY_FRAMES = 3
_stability_counters = {
    "Yaw Left": 0,
    "Yaw Right": 0,
    "Roll Left": 0,
    "Roll Right": 0,
    "Pitch Forward": 0,
    "Pitch Backward": 0,
    "Ascend/Throttle Up": 0,
    "Descend/Throttle Down": 0,
}

def stable_emit(candidate: str) -> None:
    """Emit only if the same candidate is observed for STABILITY_FRAMES consecutive frames."""
    if candidate not in _stability_counters:
        detect_and_emit(candidate)
        return
    # Increment candidate, decay others
    for key in _stability_counters.keys():
        if key == candidate:
            _stability_counters[key] = min(STABILITY_FRAMES, _stability_counters[key] + 1)
        else:
            _stability_counters[key] = max(0, _stability_counters[key] - 1)
    if _stability_counters[candidate] >= STABILITY_FRAMES:
        # reset to avoid immediate re-trigger next frame
        _stability_counters[candidate] = 0
        detect_and_emit(candidate)

def finger_direction(lmList, finger_tip, finger_base):
    """Return direction of a finger (up, down, left, right, forward, backward)."""
    x_tip, y_tip = lmList[finger_tip][1], lmList[finger_tip][2]
    x_base, y_base = lmList[finger_base][1], lmList[finger_base][2]

    dx = x_tip - x_base
    dy = y_tip - y_base

    # Up/Down check
    if abs(dy) > abs(dx):  
        if dy < -30:  # Up (y decreases upwards)
            return "up"
        elif dy > 30:
            return "down"

    # Left/Right check
    if abs(dx) > abs(dy):
        if dx < -30:
            return "left"
        elif dx > 30:
            return "right"
    return None

def is_finger_highest(lmList, finger_tip_id):
    """Check if a finger tip is higher (lower y value) than all other hand points."""
    tip_y = lmList[finger_tip_id][2]
    # Check against all 21 landmarks
    for i in range(21):
        if i == finger_tip_id:
            continue
        if lmList[i][2] < tip_y:  # Another point is higher
            return False
    return True

def is_thumb_highest(lmList):
    """Check if thumb tip (4) is higher than all other hand points."""
    return is_finger_highest(lmList, 4)

def is_thumb_lowest(lmList):
    """Check if thumb tip (4) is lower than all other hand points."""
    tip_y = lmList[4][2]
    for i in range(21):
        if i == 4:
            continue
        if lmList[i][2] > tip_y:  # Another point is lower
            return False
    return True

def are_both_fingers_highest(lmList, finger1_id, finger2_id):
    """Check if both finger tips are higher than all other hand points (excluding themselves)."""
    tip1_y = lmList[finger1_id][2]
    tip2_y = lmList[finger2_id][2]
    
    # Check against all 21 landmarks except the two finger tips
    for i in range(21):
        if i == finger1_id or i == finger2_id:
            continue
        # If any other point is higher than either finger, return False
        if lmList[i][2] < tip1_y or lmList[i][2] < tip2_y:
            return False
    return True

def is_hand_vertical(lmList):
    """Check if hand is oriented vertically (not tilted sideways for roll).
    
    Checks if the wrist-to-middle-knuckle line is more vertical than horizontal.
    This prevents roll gestures from being detected as pitch.
    """
    wrist_x, wrist_y = lmList[0][1], lmList[0][2]  # wrist
    middle_mcp_x, middle_mcp_y = lmList[9][1], lmList[9][2]  # middle finger MCP (knuckle)
    
    dx = abs(middle_mcp_x - wrist_x)
    dy = abs(middle_mcp_y - wrist_y)
    
    # Hand is vertical if vertical displacement is clearly greater than horizontal
    return dy > dx * 1.1  # tighten threshold to reduce false pitch during roll

def is_hand_horizontal(lmList):
    """Opposite of vertical; used to gate roll detection."""
    wrist_x, wrist_y = lmList[0][1], lmList[0][2]
    middle_mcp_x, middle_mcp_y = lmList[9][1], lmList[9][2]
    dx = abs(middle_mcp_x - wrist_x)
    dy = abs(middle_mcp_y - wrist_y)
    return dx > dy * 1.0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=True)

    if len(lmList) != 0:
        # Throttle: thumb tip compared to all other points
        if is_thumb_highest(lmList):
            stable_emit("Ascend/Throttle Up")
        elif is_thumb_lowest(lmList):
            stable_emit("Descend/Throttle Down")

        # Index (tip=8, base=5) and Pinky (tip=20, base=17)
        index_dir = finger_direction(lmList, 8, 5)
        pinky_dir = finger_direction(lmList, 20, 17)

        # Yaw first: require both index and pinky to move in the same horizontal direction
        if index_dir == "left" and pinky_dir == "left":
            stable_emit("Yaw Left")
        elif index_dir == "right" and pinky_dir == "right":
            stable_emit("Yaw Right")

        # Roll only when hand is horizontal, pinky is neutral, and index moves horizontally
        if is_hand_horizontal(lmList) and pinky_dir is None:
            if index_dir == "left":
                stable_emit("Roll Left")
            elif index_dir == "right":
                stable_emit("Roll Right")

        # Pitch: check finger positions (only when hand is vertical, not tilted for roll)
        if is_hand_vertical(lmList):
            index_highest = is_finger_highest(lmList, 8)  # index tip
            both_highest = are_both_fingers_highest(lmList, 8, 12)  # index and middle tips
            
            if both_highest:
                # Both index and middle are higher than all other points
                stable_emit("Pitch Backward")
            elif index_highest:
                # Only index is highest
                stable_emit("Pitch Forward")


    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (60, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(1)
