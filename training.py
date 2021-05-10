import cv2
import numpy as np
from PIL import Image
import os


# Path for face image database
path = 'data/db_face/train/wholeface'
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')  # haarcascade_frontalface_default.xml
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# function to get the images and label data
def getImagesAndLabels(path):
    dirPaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for dirPath in dirPaths:

        imagePaths = [os.path.join(dirPath, f) for f in os.listdir(dirPath)]
        id = int(os.path.split(dirPath)[-1])

        for imagePath in imagePaths:

            print("load image path : ", imagePath)

            PIL_img = Image.open(imagePath).convert('L') # grayscale

            #PIL_img.show()

            img_numpy = np.array(PIL_img,'uint8')

            faces = faceCascade.detectMultiScale(img_numpy, 1.3, 5)

            for (x,y,w,h) in faces:
                face_img = img_numpy[y:y + h, x:x + w]

                eyes = eyeCascade.detectMultiScale(
                    face_img,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(20, 20)
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
                    #cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
                    index = index + 1

                cv2.imshow('face', face_img)
                if index > 1:
                    rotatedface_img = face_alignment(face_img, eye_1, eye_2)
                    rotatedface_img = cv2.resize(rotatedface_img, (200, 200))
                    cv2.imshow('rotated', rotatedface_img)
                    faceSamples.append(rotatedface_img)
                    ids.append(id)
                cv2.waitKey()

    return faceSamples,ids

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


print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))