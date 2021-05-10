import numpy as np
import cv2
import os

def face_alignment(img, eye_position1, eye_position2):
    # face alignment based eye position
    if eye_position1 == None or eye_position2 == None:
        return -1

    if eye_position1[0] < eye_position2[0]:
        left_eye = eye_position1
        right_eye = eye_position2
    else:
        left_eye = eye_position2
        right_eye = eye_position1

    # Calculating coordinates of a central points of the rectangles
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = np.arctan(delta_y / delta_x)
    angle = (angle * 180) / np.pi

    # Width and height of the image
    h, w = img.shape[:2]
    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
    center = (w // 2, h // 2)
    # Defining a matrix M and calling
    # cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the
    # cv2.warpAffine method
    rotated_img = cv2.warpAffine(img, M, (w, h))

    return rotated_img


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')  # haarcascade_frontalface_default.xml
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()

    if frame.all() == None:
        print("frame is  None.")
        break

    color = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    '''faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        #minSize=(20, 20)
    )'''
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        index = 0
        eye_1 = None
        eye_2 = None
        for (ex, ey, ew, eh) in eyes:
            if index == 0:
                eye_1 = (ex, ey, ew, eh)
            elif index == 1:
                eye_2 = (ex, ey, ew, eh)
            # Drawing rectangles around the eyes
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
            index = index + 1

        rotated_img = face_alignment(color, eye_1, eye_2)
        cv2.imshow('rotated', rotated_img)

    cv2.imshow('frame', frame)

capture.release()
cv2.destroyAllWindows()


# http://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/