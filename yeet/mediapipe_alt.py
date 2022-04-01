import cv2
import mediapipe as mp
import serial

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
arduino = serial.Serial("COM5", 115200, timeout=1)


def send_coords(coords):
    print(f"Sending {coords[0]} {coords[1]} to arduino...")
    arduino.write(f"{coords[0]} {coords[1]}\n".encode())


try:
    cam = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cam.isOpened():
            ret, frame = cam.read()
            frame.flags.writeable = False # Can fuck around with this
            results = pose.process(frame)

            if not results.pose_landmarks:
                pass

            else:
                frame_h, frame_w, _ = frame.shape
                right_hip_x = results.pose_landmarks.landmark[24].x * frame_w
                right_hip_y = results.pose_landmarks.landmark[24].y * frame_h

                left_hip_x = results.pose_landmarks.landmark[23].x * frame_w
                left_hip_y = results.pose_landmarks.landmark[23].y * frame_h

                right_knee_x = results.pose_landmarks.landmark[26].x * frame_w
                right_knee_y = results.pose_landmarks.landmark[26].y * frame_h

                left_knee_x = results.pose_landmarks.landmark[25].x * frame_w
                left_knee_y = results.pose_landmarks.landmark[25].y * frame_h

                # Uncomment this for hip and knee circles
                # cv2.circle(frame, [int(right_hip_x), int(right_hip_y)], 3, [0, 255, 0], -1)
                # cv2.circle(frame, [int(left_hip_x), int(left_hip_y)], 3, [0, 255, 0], -1)
                # cv2.circle(frame, [int(right_knee_x), int(right_knee_y)], 3, [0, 255, 0], -1)
                # cv2.circle(frame, [int(left_knee_x), int(left_knee_y)], 3, [0, 255, 0], -1)

                # Uncomment this for lower frame skeleton
                # cv2.line(frame, [int(left_hip_x), int(left_hip_y)], [int(right_hip_x), int(right_hip_y)], [0, 0, 255], 1)
                # cv2.line(frame, [int(left_knee_x), int(left_knee_y)], [int(left_hip_x), int(left_hip_y)], [0, 0, 255], 1)
                # cv2.line(frame, [int(right_knee_x), int(right_knee_y)], [int(right_hip_x), int(right_hip_y)], [0, 0, 255], 1)

                x = int((left_hip_x + right_hip_x) / 2)
                y = int(left_hip_y + 10)

                # Uncomment this to put circle on dick
                # cv2.circle(frame, [x, y], 5, [0, 255, 0], -1)

                send_coords([x, y])

            # Uncomment this for full skeleton drawing
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            #                           landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Uncomment this to open window
            # cv2.imshow("Cock detection", cv2.flip(frame, 1))
            # if cv2.waitKey(10) & 0xFF == ord("q"):
            #     break

except KeyboardInterrupt:
    cam.release()
    cv2.destroyAllWindows()
