"""
========================================================
REAL-TIME MOTION COMBAT POSE DETECTION
========================================================

POSE GUIDE
--------------------------------------------------------

IDLE
- Stand naturally.
- No action is triggered.

SHIELD
- Perform a squat/crouch.
- Hips move closer to knees.
- Torso should remain relatively upright.

JUMP
- Jump vertically.
- Hips and ankles must rise above their standing position.

DODGE LEFT
- Move your entire body to the left.
- Hip center must move beyond the dodge threshold.

DODGE RIGHT
- Move your entire body to the right.
- Hip center must move beyond the dodge threshold.

SHOOT (LEFT-HANDED)
- Left arm extended forward.
- Right wrist pulled close to right shoulder.
- Both conditions must be true simultaneously.

SHOOT (RIGHT-HANDED)
- Right arm extended forward.
- Left wrist pulled close to left shoulder.
- Both conditions must be true simultaneously.

========================================================
IMPLEMENTED FEATURES
========================================================

[✓] Real-Time Pose Tracking
[✓] Shield Detection
[✓] Jump Detection
[✓] Dodge Left Detection
[✓] Dodge Right Detection
[✓] Bow Attack Detection
[✓] Left-Handed Support
[✓] Right-Handed Support
[✓] Real-Time Action Display

========================================================
TECHNOLOGIES USED
========================================================

- Python
- OpenCV
- MediaPipe Pose Estimation

========================================================
PROJECT TITLE
========================================================

Real-Time Motion-Based Combat Pose Detection System

========================================================
AUTHOR
========================================================

SACCHIDANAND
2026

========================================================
"""
import cv2
import mediapipe as mp
import time

# ==========================================
# MediaPipe Setup
# ==========================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# Camera Setup
# ==========================================

cap = cv2.VideoCapture(0)

# ==========================================
# State Variables
# ==========================================

current_action = "IDLE"

# When a dodge happens, keep the text visible
# until this time is reached.
action_end_time = 0

# Dynamic center position used for dodge detection.
center_x = None
neutral_hip_y = None
neutral_ankle_y = None
jump_cooldown = False


# ==========================================
# Pose Detection
# ==========================================

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Convert image to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        now = time.time()

        # --------------------------------------
        # Reset dodge text after timer expires
        # --------------------------------------

        if (
            current_action in ["DODGE LEFT", "DODGE RIGHT"]
            and now > action_end_time
        ):
            current_action = "IDLE"

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # ==================================
            # HIP LANDMARKS
            # ==================================

            left_hip = landmarks[
                mp_pose.PoseLandmark.LEFT_HIP
            ]

            right_hip = landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP
            ]

            # ==================================
            # ANKLE LANDMARKS
            # ==================================

            left_ankle = landmarks[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ]

            right_ankle = landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ]
            # WRISTS

            left_wrist = landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST
            ]

            right_wrist = landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]
            left_shoulder = landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ]

            right_shoulder = landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]

            # SHOULDERS

            left_shoulder = landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ]

            right_shoulder = landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]

            # ==================================
            # BODY CENTER
            # ==================================

            hip_center_x = (
                left_hip.x + right_hip.x
            ) / 2

            current_hip_y = (
                left_hip.y + right_hip.y
            ) / 2

            current_ankle_y = (
                left_ankle.y + right_ankle.y
            ) / 2

            # First detected position becomes center
            if center_x is None:
                center_x = hip_center_x

            if neutral_hip_y is None:
                neutral_hip_y = current_hip_y

            if neutral_ankle_y is None:
                neutral_ankle_y = current_ankle_y

            # ==================================
            # SHIELD DETECTION
            # ==================================

            left_knee = landmarks[
                mp_pose.PoseLandmark.LEFT_KNEE
            ]

            right_knee = landmarks[
                mp_pose.PoseLandmark.RIGHT_KNEE
            ]

            left_shoulder = landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ]

            right_shoulder = landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]

            hip_y = (left_hip.y + right_hip.y) / 2

            knee_y = (left_knee.y + right_knee.y) / 2

            shoulder_y = (
                left_shoulder.y + right_shoulder.y
            ) / 2

            hip_knee_distance = abs(
                knee_y - hip_y
            )

            torso_distance = abs(
                shoulder_y - hip_y
            )

            shield_active = (
                hip_knee_distance < 0.12
                and
                torso_distance > 0.18
            )

            # ==================================
            # JUMP DETECTION
            # ==================================

            hip_jump = (
                neutral_hip_y - current_hip_y
            ) > 0.03

            ankle_jump = (
                neutral_ankle_y - current_ankle_y
            ) > 0.03

            jump_active = (
                hip_jump and ankle_jump
            )

            # ==================================
            # BOW DETECTION
            # ==================================

            # Left-handed bow
            left_arm_extended = (
                abs(left_wrist.x - left_shoulder.x) > 0.15
            )

            right_hand_loaded = (
                abs(right_wrist.x - right_shoulder.x) < 0.05
                and
                abs(right_wrist.y - right_shoulder.y) < 0.08
            )

            # Right-handed bow
            right_arm_extended = (
                abs(right_wrist.x - right_shoulder.x) > 0.05
            )

            left_hand_loaded = (
                abs(left_wrist.x - left_shoulder.x) < 0.05
                and
                abs(left_wrist.y - left_shoulder.y) < 0.08
            )

            bow_pose = (

                (left_arm_extended and right_hand_loaded)

                or

                (right_arm_extended and left_hand_loaded)

            )
            

            

            # ==================================
            # DODGE DETECTION
            # ==================================

            movement_x = hip_center_x - center_x

            # Increase this if dodges trigger too easily.
            dodge_threshold = 0.20

            # Shield has priority over dodge.
            if shield_active:

                current_action = "SHIELD"

            elif jump_active:

                current_action = "JUMP"

            elif bow_pose:

                current_action = "SHOOT"

            else:

                # Leaving shield returns to idle.
                if current_action == "SHIELD":
                    current_action = "IDLE"

                if current_action == "SHOOT":
                    current_action = "IDLE"

                # ----------------------------------
                # DODGE LEFT
                #
                # Camera is mirrored, so movement
                # directions are intentionally
                # inverted for the player.
                # ----------------------------------

                if movement_x > dodge_threshold:

                    current_action = "DODGE LEFT"

                    action_end_time = now + 3

                    # New position becomes center.
                    center_x = hip_center_x

                # ----------------------------------
                # DODGE RIGHT
                # ----------------------------------

                elif movement_x < -dodge_threshold:

                    current_action = "DODGE RIGHT"

                    action_end_time = now + 3

                    center_x = hip_center_x

            # ==================================
            # Draw Pose Skeleton
            # ==================================

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # ======================================
        # Text Styling
        # ======================================

        if current_action == "IDLE":
            color = (255, 255, 255)  # White
        else:
            color = (0, 0, 255)      # Red

        cv2.putText(
            frame,
            f"Action: {current_action}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3
        )

        cv2.imshow(
            "Combat Pose Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()