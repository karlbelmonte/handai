import collections
import math
import os
import time
import urllib.request
import serial
import serial.tools.list_ports
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def get_hand_label(handedness):
    if not handedness:
        return "Unknown"
    return handedness[0].category_name


def distance_3d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def hand_normal(landmarks):
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z)
    v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z)
    normal = (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )
    length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if length == 0:
        return 0.0, 0.0, 0.0
    return normal[0] / length, normal[1] / length, normal[2] / length


def get_finger_mcp_alignment(landmarks, is_left_hand=False):
    wrist = landmarks[0]
    thumb_cmc = landmarks[1]
    
    dx = thumb_cmc.x - wrist.x
    dy = thumb_cmc.y - wrist.y
    
    if is_left_hand:
        dx = -dx
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    raw_angle = -angle_deg
    
    FLAT_RAW = 30.0
    STRAIGHT_RAW = 90.0
    
    calibrated_angle = (raw_angle - FLAT_RAW) * 90.0 / (STRAIGHT_RAW - FLAT_RAW)
    calibrated_angle = max(0, min(90, calibrated_angle))
    
    return round(calibrated_angle, 1)


def is_thumb_overlapping_palm(landmarks):
    """
    Detect if thumb tip overlaps/touches ANY finger node (closed hand).
    Returns True when thumb tip touches ANY finger (index/middle/ring/pinky tip or MCP).
    Returns False when thumb is away from all fingers (open hand).
    """
    thumb_tip = landmarks[4]
    
    # All finger nodes to check: tips (8,12,16,20) and MCP knuckles (5,9,13,17)
    finger_nodes = [
        landmarks[8],   # Index tip
        landmarks[12],  # Middle tip
        landmarks[16],  # Ring tip
        landmarks[20],  # Pinky tip
        landmarks[5],   # Index MCP
        landmarks[9],   # Middle MCP
        landmarks[13],  # Ring MCP
        landmarks[17],  # Pinky MCP
    ]
    
    # Distance threshold: thumb tip touches if within 0.05 (5% of screen)
    TOUCH_THRESHOLD = 0.05
    
    for finger in finger_nodes:
        dist = math.sqrt(
            (thumb_tip.x - finger.x) ** 2 +
            (thumb_tip.y - finger.y) ** 2 +
            (thumb_tip.z - finger.z) ** 2
        )
        if dist < TOUCH_THRESHOLD:
            return True  # Thumb touches this finger
    
    return False  # Thumb doesn't touch any finger


def get_hand_pose(landmarks, is_left_hand=False):
    index_mcp = landmarks[5]
    index_tip = landmarks[8]
    middle_mcp = landmarks[9]
    middle_tip = landmarks[12]
    ring_mcp = landmarks[13]
    ring_tip = landmarks[16]
    pinky_mcp = landmarks[17]
    pinky_tip = landmarks[20]

    index_up = index_tip.y < index_mcp.y
    middle_up = middle_tip.y < middle_mcp.y
    ring_up = ring_tip.y < ring_mcp.y
    pinky_up = pinky_tip.y < pinky_mcp.y

    angle = get_finger_mcp_alignment(landmarks, is_left_hand)
    
    # Check if thumb tip overlaps ANY finger node
    thumb_overlaps = is_thumb_overlapping_palm(landmarks)

    # "Closed" when thumb touches any finger, "Open" when thumb is away
    pose_parts = ["Closed" if thumb_overlaps else "Open"]

    is_pointing = index_up and not middle_up and not ring_up and not pinky_up
    if is_pointing:
        pose_parts.append("Wrist")

    return ", ".join(pose_parts), angle, thumb_overlaps, is_pointing


def main():
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print(f"Downloading model...")
        url = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    ser_right = None  # COM3 - Right Hand (A-F)
    ser_left = None   # COM4 - Left Hand (G-L)
    
    try:
        ser_right = serial.Serial('COM3', 115200, timeout=1)
        print(f"Connected to RIGHT Arduino on COM3")
        time.sleep(2)
    except Exception as e:
        print(f"Serial Error (COM3): {e}")
    
    try:
        ser_left = serial.Serial('COM4', 115200, timeout=1)
        print(f"Connected to LEFT Arduino on COM4")
        time.sleep(2)
    except Exception as e:
        print(f"Serial Error (COM4): {e}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Person Perspective Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Person Perspective Tracker", 1280, 720)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_queue = collections.deque()
        last_processed_result = None
        frame_counter = 0
        
        right_hand_open = None
        left_hand_open = None
        right_wrist_active = False
        left_wrist_active = False
        
        last_A = None
        last_B = None
        last_C = None
        last_D = None
        last_E = None
        last_F = None
        
        last_G = None
        last_H = None
        last_I = None
        last_J = None
        last_K = None
        last_L = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            process_frame = frame.copy()
            display_frame = cv2.flip(frame, 1)

            now = time.time()
            frame_queue.append((now, process_frame))

            if len(frame_queue) > 10:
                frame_queue.popleft()

            if frame_queue and now - frame_queue[0][0] >= 0.05:
                frame_time, process_frame = frame_queue.popleft()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                    data=cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB))
                last_processed_result = landmarker.detect_for_video(mp_image, int(frame_time * 1000))

            annotated_frame = display_frame.copy()
            h, w, _ = annotated_frame.shape
            
            right_hand_coords = (0, 0, 0)
            right_hand_angle = 0
            left_hand_coords = (0, 0, 0)
            left_hand_angle = 0

            if last_processed_result and last_processed_result.hand_landmarks:
                for i, landmarks in enumerate(last_processed_result.hand_landmarks):
                    label = get_hand_label(last_processed_result.handedness[i])
                    color = (0, 255, 0) if label == "Right" else (255, 0, 0)
                    
                    wrist = landmarks[0]
                    
                    x_centered = int((0.5 - wrist.x) * 180)
                    y_centered = int((0.5 - wrist.y) * 180)
                    
                    index_mcp = landmarks[5]
                    middle_mcp = landmarks[9]
                    
                    dist_index_wrist = math.sqrt(
                        (index_mcp.x - wrist.x) ** 2 +
                        (index_mcp.y - wrist.y) ** 2 +
                        (index_mcp.z - wrist.z) ** 2
                    )
                    dist_middle_wrist = math.sqrt(
                        (middle_mcp.x - wrist.x) ** 2 +
                        (middle_mcp.y - wrist.y) ** 2 +
                        (middle_mcp.z - wrist.z) ** 2
                    )
                    
                    knuckle_wrist_avg = (dist_index_wrist + dist_middle_wrist) / 2
                    raw_z = int(knuckle_wrist_avg * 1000)

                    z_centered = int((raw_z - 300) * 60 / 100)
                    z_centered = max(0, min(90, z_centered))
                    
                    is_left = label == "Left"
                    pose_str, angle, thumb_overlaps, is_pointing = get_hand_pose(landmarks, is_left)
                    
                    hand_status = "CLOSED" if thumb_overlaps else "OPEN"
                    if label == "Right":
                        right_hand_open = hand_status
                        right_wrist_active = is_pointing
                    else:
                        left_hand_open = hand_status
                        left_wrist_active = is_pointing
                    
                    if label == "Right":
                        right_hand_coords = (x_centered, y_centered, z_centered)
                        right_hand_angle = angle
                        wrist_pixel_x = int(wrist.x * w)
                        wrist_pixel_y = int(wrist.y * h)
                        cv2.circle(annotated_frame, (wrist_pixel_x, wrist_pixel_y), 8, (0, 255, 0), -1)
                    else:
                        left_hand_coords = (x_centered, y_centered, z_centered)
                        left_hand_angle = angle
                        wrist_pixel_x = int(wrist.x * w)
                        wrist_pixel_y = int(wrist.y * h)
                        cv2.circle(annotated_frame, (wrist_pixel_x, wrist_pixel_y), 8, (255, 0, 0), -1)
                    
                    for lm in landmarks:
                        cx = int((1.0 - lm.x) * w)
                        cy = int(lm.y * h)
                        cv2.circle(annotated_frame, (cx, cy), 4, color, -1)

            ui_x = 20
            right_y = 40
            right_angle_y = 75
            left_y = 110
            left_angle_y = 140
            
            x, y, z = right_hand_coords
            cv2.putText(annotated_frame, f"RIGHT X: {x} Y: {y} Z: {z}", (ui_x, right_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"RIGHT X: {x} Y: {y} Z: {z}", (ui_x, right_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"ANGLE: {right_hand_angle} deg", (ui_x, right_angle_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"ANGLE: {right_hand_angle} deg", (ui_x, right_angle_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            x, y, z = left_hand_coords
            cv2.putText(annotated_frame, f"LEFT X: {x} Y: {y} Z: {z}", (ui_x, left_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"LEFT X: {x} Y: {y} Z: {z}", (ui_x, left_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"ANGLE: {left_hand_angle} deg", (ui_x, left_angle_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"ANGLE: {left_hand_angle} deg", (ui_x, left_angle_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            right_ui_x = w - 200
            status_y1 = 40
            status_y2 = 75
            
            right_prefix = "[WRIST] " if right_wrist_active else ""
            right_status_text = f"RIGHT: {right_prefix}{right_hand_open or '---'}"
            right_status_color = (0, 255, 0) if right_hand_open == "OPEN" else (0, 0, 255)
            cv2.putText(annotated_frame, right_status_text, (right_ui_x, status_y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated_frame, right_status_text, (right_ui_x, status_y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_status_color, 2, cv2.LINE_AA)
            
            left_prefix = "[WRIST] " if left_wrist_active else ""
            left_status_text = f"LEFT: {left_prefix}{left_hand_open or '---'}"
            left_status_color = (255, 0, 0) if left_hand_open == "OPEN" else (0, 0, 255)
            cv2.putText(annotated_frame, left_status_text, (right_ui_x, status_y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated_frame, left_status_text, (right_ui_x, status_y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_status_color, 2, cv2.LINE_AA)

            frame_counter += 1
            if frame_counter % 8 == 0:
                rx, ry, rz = right_hand_coords
                lx, ly, lz = left_hand_coords
                right_open_val = 1 if right_hand_open == "OPEN" else 0
                left_open_val = 1 if left_hand_open == "OPEN" else 0
                
                if ser_right and ser_right.is_open:
                    try:
                        if rz != last_A:
                            ser_right.write(f"A:{rz}\n".encode())
                            last_A = rz
                        
                        if ry != last_B:
                            ser_right.write(f"B:{ry}\n".encode())
                            last_B = ry
                        
                        if rx != last_C:
                            ser_right.write(f"C:{rx}\n".encode())
                            last_C = rx
                        
                        if right_wrist_active and rx != last_D:
                            ser_right.write(f"D:{rx}\n".encode())
                            last_D = rx
                        
                        if right_hand_angle != last_E:
                            ser_right.write(f"E:{right_hand_angle}\n".encode())
                            last_E = right_hand_angle
                        
                        if right_open_val != last_F:
                            ser_right.write(f"F:{right_open_val}\n".encode())
                            last_F = right_open_val
                    except Exception as e:
                        print(f"Serial write error (COM3): {e}")
                
                if ser_left and ser_left.is_open:
                    try:
                        if lz != last_G:
                            ser_left.write(f"G:{lz}\n".encode())
                            last_G = lz
                        
                        if ly != last_H:
                            ser_left.write(f"H:{ly}\n".encode())
                            last_H = ly
                        
                        if lx != last_I:
                            ser_left.write(f"I:{lx}\n".encode())
                            last_I = lx
                        
                        if left_wrist_active and lx != last_J:
                            ser_left.write(f"J:{lx}\n".encode())
                            last_J = lx
                        
                        if left_hand_angle != last_K:
                            ser_left.write(f"K:{left_hand_angle}\n".encode())
                            last_K = left_hand_angle
                        
                        if left_open_val != last_L:
                            ser_left.write(f"L:{left_open_val}\n".encode())
                            last_L = left_open_val
                    except Exception as e:
                        print(f"Serial write error (COM4): {e}")

            cv2.imshow("Person Perspective Tracker", annotated_frame)
            time.sleep(0.016)
            
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]: 
                break

    cap.release()
    cv2.destroyAllWindows()
    if ser_right:
        ser_right.close()
    if ser_left:
        ser_left.close()


if __name__ == "__main__":
    main()