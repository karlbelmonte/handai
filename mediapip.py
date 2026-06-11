import collections
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def get_hand_label(handedness):       
    if not handedness:
        return "Unknown"
    # Returns 'Left' or 'Right'
    return handedness[0].category_name

def distance_3d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def hand_normal(landmarks):
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z)
    v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z)

    # Cross product to find the palm normal vector
    normal = (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )

    length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if length == 0:
        return 0.0, 0.0, 0.0

    return normal[0] / length, normal[1] / length, normal[2] / length

def get_hand_pose(landmarks):

    # Thumb tip and pinky tip
    thumb_tip = landmarks[4]
    pinky_tip = landmarks[20]

    # Distance between thumb and pinky
    thumb_pinky_distance = distance_3d(thumb_tip, pinky_tip)

    # Smaller distance = closed hand
    is_closed = thumb_pinky_distance < 0.18

    # -----------------------------
    # TWIST CHECK
    # -----------------------------

    normal_x, _, _ = hand_normal(landmarks)

    is_twisted = abs(normal_x) > 0.65

    pose_parts = []

    if is_closed:
        pose_parts.append("Closed")
    else:
        pose_parts.append("Open")

    if is_twisted:
        pose_parts.append("Twisted")

    return ", ".join(pose_parts)

def main():
    model_path = "hand_landmarker.task"
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        print(f"Downloading hand_landmarker.task...")
        url = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded successfully!")
        except Exception as e:
            print(f"ERROR: Failed to download model from {url}")
            print(f"Exception: {e}")
            print(f"\nPlease manually download the hand_landmarker.task from:")
            print(f"https://developers.google.com/mediapipe/solutions/vision/hand_landmarker")
            print(f"\nOr place hand_landmarker.task in the current directory")
            return

    cap = cv2.VideoCapture(0)

    # Initialize Hand Landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_queue = collections.deque()
        last_processed_result = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            
            now = time.time()
            frame_queue.append((now, frame.copy()))

            # Maintain a small buffer for smooth processing
            if frame_queue and now - frame_queue[0][0] >= 0.05:
                frame_time, process_frame = frame_queue.popleft()
                
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, 
                    data=cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                )
                
                # Detect hands in the video frame
                last_processed_result = landmarker.detect_for_video(mp_image, int(frame_time * 1000))

            annotated_frame = frame.copy()
            h, w, _ = annotated_frame.shape

            if last_processed_result and last_processed_result.hand_landmarks:
                for i, landmarks in enumerate(last_processed_result.hand_landmarks):
                    # Get Hand Label (Left/Right)
                    label = "Unknown"
                    if i < len(last_processed_result.handedness):
                        label = get_hand_label(last_processed_result.handedness[i])

                    # Get Pose (This is where "Closed, Twisted" is built)
                    pose_str = get_hand_pose(landmarks)

                    # Print pose information to terminal
                    full_display = f"{label}: {pose_str}"
                    print(full_display)

                    # UI Settings
                    color = (0, 255, 0) if "Right" in label else (255, 0, 0)
                    text_y = 60 + (i * 45)

                    # Draw text with shadow for better readability on camera
                    cv2.putText(annotated_frame, full_display, (20, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(annotated_frame, full_display, (20, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                    # Draw hand skeletons
                    for lm in landmarks:
                        cv2.circle(annotated_frame, (int(lm.x * w), int(lm.y * h)), 3, color, -1)

            cv2.imshow("Hand Pose Tracker", annotated_frame)
            
            # Press 'q' or 'Esc' to exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()