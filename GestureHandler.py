
import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Settings
CAMERA_INDEX = 0
MODEL_PATH = "hand_landmarker.task"

VIDEO_X = 960
VIDEO_Y = 540

SWIPE_DISTANCE = 0.12
COOLDOWN = 1.0
FIST_HOLD = 2.0

pyautogui.FAILSAFE = True

# New MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_up(hand, tip, joint):
    return hand[tip].y < hand[joint].y


def recognize_gesture(hand):
    index = finger_up(hand, 8, 6)
    middle = finger_up(hand, 12, 10)
    ring = finger_up(hand, 16, 14)
    pinky = finger_up(hand, 20, 18)

    thumb = distance(hand[4], hand[5]) > distance(hand[3], hand[5])

    if index and not middle and not ring and not pinky:
        return "point"

    if index and middle and ring and pinky and thumb:
        return "open"

    if not index and not middle and not ring and not pinky:
        return "fist"

    return "other"


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Cannot open webcam.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    previous_x = None
    previous_time = None
    last_action = 0
    fist_start = None
    fist_done = False
    pointing = False
    start_time = time.monotonic()

    try:
        with HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = camera.read()

                if not success:
                    break

                # Mirror webcam view
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(
                    (time.monotonic() - start_time) * 1000
                )

                result = detector.detect_for_video(image, timestamp)
                status = "Show your hand"
                gesture = "none"

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    gesture = recognize_gesture(hand)

                    # Approximate hand position using wrist landmark
                    x = hand[0].x
                    y = hand[0].y
                    now = time.monotonic()

                    # Draw hand landmarks
                    h, w = frame.shape[:2]
                    for landmark in hand:
                        px = int(landmark.x * w)
                        py = int(landmark.y * h)
                        cv2.circle(
                            frame, (px, py), 4, (0, 255, 0), -1
                        )
                        # Open-hand swipe left or right
# Open-hand swipe left or right
                    if (
                        gesture == "point"
                        and not pointing
                        and now - last_action >= COOLDOWN
                    ):
                        pyautogui.click(VIDEO_X, VIDEO_Y)
                        pyautogui.PAUSE=5
                        status = "VIDEO CLICK"
                        last_action = now

                    pointing = gesture == "point"

                    # Hold fist in the center to close PowerPoint
                    in_center = (
                        0.35 <= x <= 0.65
                        and 0.25 <= y <= 0.75
                    )

                    if gesture == "fist" and in_center:
                        if fist_start is None:
                            fist_start = now

                        remaining = FIST_HOLD - (now - fist_start)
                        status = f"Hold fist: {max(0, remaining):.1f}s"

                        if remaining <= 0 and not fist_done:
                            pyautogui.hotkey("alt", "f4")
                            status = "CLOSE POWERPOINT"
                            fist_done = True
                            last_action = now
                    else:
                        fist_start = None
                        fist_done = False

                    if gesture == "open":
                        status = status if status in (
                            "NEXT SLIDE", "PREVIOUS SLIDE"
                        ) else "Open hand: swipe"
                    elif gesture == "point":
                        status = "Point: video click"
                    elif gesture == "fist" and not in_center:
                        status = "Move fist to center and hold"

                else:
                    previous_x = None
                    previous_time = None
                    fist_start = None
                    fist_done = False
                    pointing = False

                cv2.putText(
                    frame, "Gesture: " + gesture,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2
                )
                cv2.putText(
                    frame, "Press q to quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )

                cv2.imshow("Slide Gesture Controls", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


