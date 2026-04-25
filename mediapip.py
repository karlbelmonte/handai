import argparse
import os
import sys
import time

import cv2
import mediapipe as mp

# Aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def get_hand_label(handedness):
    if not handedness:
        return "Unknown"
    category = handedness[0]
    return category.category_name or category.display_name or "Unknown"


def print_result(result):
    if not result.hand_landmarks:
        print("No hands detected")
        return

    for hand_index, landmarks in enumerate(result.hand_landmarks):
        label = get_hand_label(result.handedness[hand_index] if len(result.handedness) > hand_index else None)
        print(f"Hand {hand_index + 1}: {label}")
        for landmark_index, landmark in enumerate(landmarks):
            print(
                f"  Landmark {landmark_index:02d}: "
                f"x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}"
            )


def create_landmarker(model_path):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )
    return HandLandmarker.create_from_options(options)


def resolve_model_path(model_path):
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Run MediaPipe hand landmarker on camera or video input.")
    parser.add_argument("--model", default="hand_landmarker.task", help="Path to the hand landmarker task file.")
    parser.add_argument("--input", default=None, help="Optional video file input. If omitted, uses camera 0.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: model file not found: {model_path}")
        print("Download or move hand_landmarker.task into the workspace and pass it with --model.")
        sys.exit(1)

    source = args.input if args.input else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open video source: {source}")
        if args.input is None:
            print("If you are in a container or VM without a camera, pass --input <video.mp4>.")
        sys.exit(1)

    try:
        with create_landmarker(model_path) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect(mp_image)

                print_result(result)

                height, width, _ = frame.shape
                annotated_frame = frame.copy()
                if result.hand_landmarks:
                    for hand_index, landmarks in enumerate(result.hand_landmarks):
                        label = get_hand_label(result.handedness[hand_index] if len(result.handedness) > hand_index else None)
                        color = (0, 255, 0) if label.lower() == "right" else (255, 0, 0)
                        cv2.putText(
                            annotated_frame,
                            f"{label} hand {hand_index + 1}",
                            (10, 30 + hand_index * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )
                        for landmark in landmarks:
                            x_px = int(landmark.x * width)
                            y_px = int(landmark.y * height)
                            cv2.circle(annotated_frame, (x_px, y_px), 3, color, -1)

                cv2.imshow("Camera", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except OSError as exc:
        print("ERROR: failed to load MediaPipe native library.")
        print(exc)
        print("Install missing OpenGL support: sudo apt update && sudo apt install libgles2-mesa")
        sys.exit(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
