import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')  # haarcascade_frontalface_default.xml
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

font = cv2.FONT_HERSHEY_SIMPLEX


def face_alignment(img, eye_position1, eye_position2):
    # face alignment based eye position
    #if eye_position1 == None or eye_position2 == None:
    #    return -1

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

    if delta_x == 0 :
        delta_x = 1

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




# iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['none', 'mira', 'jihyun', 'inseong', 'hyunbin', 'jiwon', 'obama', 'son', 'jimi']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()

    if ret == False:
        continue

    #img = cv2.flip(img, -1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_img = gray[y:y + h, x:x + w]

        resizedface_img = cv2.resize(face_img, (300, 300))
        eyes = eyeCascade.detectMultiScale(
            resizedface_img,
            scaleFactor=1.1,
            minNeighbors=3
        )

        index = 0
        eye_1 = None
        eye_2 = None
        eye_cnt = len(eyes)
        if eye_cnt > 2:
            eye_1 = eyes[0]
            eye_2 = eyes[1]
            # Drawing rectangles around the eyes
            # cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
            #cv2.imshow('face', face_img)
            rotatedface_img = face_alignment(face_img, eye_1, eye_2)
            rotatedface_img = cv2.resize(rotatedface_img, (200, 200))
        else:
            rotatedface_img = cv2.resize(face_img, (200, 200))

        #cv2.imshow('rotated', rotatedface_img)
        #cv2.waitKey()

        ###########################################################
        id, confidence = recognizer.predict(rotatedface_img)
        ###########################################################

        if confidence < 500:
            confidence = int(100 * (1 - (confidence) / 300))

        # If confidence is less them 100 ==> "0" : perfect match
        if (confidence > 60):
            id = names[id]
            confidence = "  {0}%".format(confidence)
        else:
            id = "unknown"
            confidence = "  {0}%".format(confidence)

        '''if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100-confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100-confidence))'''

        cv2.putText(
            img,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()