import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2


def locate_penis(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    left_hip_x = shaped[0][1]
    left_hip_y = shaped[0][0]
    left_hip_conf = shaped[0][2]

    right_hip_x = shaped[1][1]
    right_hip_y = shaped[1][0]
    right_hip_conf = shaped[1][2]

    left_knee_x = shaped[2][1]
    left_knee_y = shaped[2][0]
    left_knee_conf = shaped[2][2]

    right_knee_x = shaped[3][1]
    right_knee_y = shaped[3][0]
    right_knee_conf = shaped[3][2]

    if left_knee_conf > confidence_threshold and right_knee_conf > confidence_threshold and left_hip_conf > confidence_threshold and right_hip_conf > confidence_threshold:
        x = int((left_hip_x + right_hip_x) / 2)
        y = int((left_hip_y + left_knee_y) / 2 - 20)
        cv2.circle(frame, [x, y], 5, [255, 0, 0], -1)

    return [x, y]


interpreter = tf.lite.Interpreter(model_path=r"resources/lite-model_movenet_singlepose_lightning_3.tflite")
interpreter.allocate_tensors()
capture = cv2.VideoCapture(0)
while capture.isOpened():
    ret, frame = capture.read()
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.float32)
    input_dets = interpreter.get_input_details()
    output_dets = interpreter.get_output_details()
    interpreter.set_tensor(input_dets[0]["index"], np.array(input_img))
    interpreter.invoke()
    keypoints_w_scores = interpreter.get_tensor(output_dets[0]["index"])
    crotch_points = [keypoints_w_scores[0][0][11], keypoints_w_scores[0][0][12], keypoints_w_scores[0][0][13], keypoints_w_scores[0][0][14]]
    coords = locate_penis(frame, crotch_points, 0.2)
    print(coords)
    frame_flipped = cv2.flip(frame, 1)
    cv2.imshow("Cock Detection", frame_flipped)

    if cv2.waitKey(10) & 0xFF==ord("q"):
        break

capture.release()
cv2.destroyAllWindows()