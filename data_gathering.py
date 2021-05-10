import numpy as np
import cv2
import os

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')  # haarcascade_frontalface_default.xml

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
count = 0
while True:
    ret, frame = capture.read()

    if frame.all() == None:
        print("frame is  None.")
        break

    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colorimg = frame.copy()

    faces = faceCascade.detectMultiScale(grayimg, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey()

    if len(faces) > 0:
        print("\n [INFO] If you want save a image, prees the 's' key ...")

        if key == ord('s') or key == ord('S'):

            directory = 'data/db_face/train/wholeface/' + face_id
            if not os.path.exists(directory):
                os.makedirs(directory)
                print('the Folder was created..: ', directory)

            filename = directory + ('/%03d' % (count)) + '.jpg'
            print(filename)
            cv2.imwrite(filename, colorimg)
            count += 1

            '''for (x, y, w, h) in faces:
                roi_color = colorimg[:][y:y + h, x:x + w]
                directory = 'data/db_face/' + face_id

                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print('the Folder was created..: ', directory)

                filename = directory + ('/%03d' % (count)) + '.jpg'
                print(filename)
                cv2.imwrite(filename, roi_color)
                count += 1
            '''

    print(key)
    if key & 0xff == 27:
        break

capture.release()
cv2.destroyAllWindows()