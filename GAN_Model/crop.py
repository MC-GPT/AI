import math
import ntpath
import tensorflow as tf
from retinaface import RetinaFace
import numpy as np
from PIL import Image
import cv2

def crop_image(input_path, output_folder):
    cpus = tf.config.experimental.list_physical_devices('cpu')
    for cpu in cpus:
        tf.config.experimental.set_memory_growth(cpu, True)

    resp = RetinaFace.detect_faces(input_path)

    img = cv2.imread(input_path)
    img = img[:, :, ::-1]

    facial_area = resp['face_1']['facial_area']
    landmarks = resp['face_1']["landmarks"]
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose = landmarks["nose"]

    face_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

    # swap
    left_eye, right_eye = right_eye, left_eye

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    center_eyes = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = np.linalg.norm(np.array(left_eye) - np.array(point_3rd))
    b = np.linalg.norm(np.array(right_eye) - np.array(point_3rd))
    c = np.linalg.norm(np.array(right_eye) - np.array(left_eye))

    if b != 0 and c != 0:
        cos_a = (b * b + c * c - a * a) / (2 * b * c)

        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        rotated_img = Image.fromarray(img)
        rotated_img = np.array(rotated_img.rotate(direction * angle, center=nose, resample=Image.Resampling.BILINEAR))

        if center_eyes[1] > nose[1]:
            rotated_img = Image.fromarray(rotated_img)
            rotated_img = np.array(rotated_img.rotate(180))

        w, h = facial_area[2] - facial_area[0], facial_area[3] - facial_area[1]
        cx, cy = (facial_area[0] + facial_area[2]) / 2, (facial_area[1] + facial_area[3]) / 1.8

        margin_ratio = 1.5

        crop_size = max(w, h) * margin_ratio

        x1 = max(0, int(cx - crop_size / 2))
        x2 = int(cx + crop_size / 2)
        y1 = max(0, int(cy - crop_size * 0.6))
        y2 = int(cy + crop_size * 0.4)

        face_img = rotated_img[y1:y2, x1:x2]

        # Save cropped image
        cropped_image = Image.fromarray(face_img)
        cropped_image.save(f'{output_folder}/crop_{ntpath.basename(input_path)}')

