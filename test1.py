

import cv2
import numpy as np
import os
import time
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from PIL import Image, ImageFont, ImageDraw


import time

import argparse
import numpy as np
import imutils
import cv2
import dlib

global finally_done
finally_done = False


global is_time
is_time = False
# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

global face_is_right
face_is_right = False
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = {0:'None', 1:'Suchith',2: 'Paula', 3:'Ilza', 4:'Z', 5:'Kaushik'}

blink_pattern = {'None':'lrl', 'None':'lll','Suchith': 'rrl', 'Paula':'llr', 'Ilza':'lrl','Kaushik':'rrr'}
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            x = round(100 - confidence)
            confidence = "  {0}%".format(round(100 - confidence))
            # print(x)

            if (x > 50):
                print('nice')
                is_time = True
                face_is_right = True


                def eye_aspect_ratio(eye):
                    A = dist.euclidean(eye[1], eye[5])
                    B = dist.euclidean(eye[2], eye[4])

                    C = dist.euclidean(eye[0], eye[3])

                    ear = (A + B) / (2.0 * C)
                    return ear


                ap = argparse.ArgumentParser()
                ap.add_argument('-p', '--shape-predictor', required=True, help='path to facial landmark predictor')
                ap.add_argument('-v', '--video', type=str, default="", help='path to input video file')
                args = vars(ap.parse_args())

                EYE_AR_THRESH = 0.2
                EYE_AR_CONSEC_FRAMES = 3

                left_COUNTER = 0
                right_COUNTER = 0
                blink_sequence = []
                left_TOTAL = 0
                right_TOTAL = 0
                global s
                s = 'not started'

                print('[INFO] Loading facial landmark predictor...')
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor(args['shape_predictor'])

                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

                print('[INFO] Starting video stream thread...')
                fileStream = False
                if args['video']:
                    vs = FileVideoStream(args['video']).start()
                    fileStream = True
                else:
                    vs = VideoStream(src=0).start()
                    # vs = VideoStream(usePiCamera=True).start()
                    fileStream = False

                time.sleep(1.0)

                while True:
                    if fileStream and not vs.more():
                        break

                    frame = vs.read()
                    frame = imutils.resize(frame, width=450)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    rects = detector(gray, 0)

                    for rect in rects:
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        leftEye = shape[lStart: lEnd]
                        rightEye = shape[rStart: rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)

                        ear = (leftEAR + rightEAR) / 2.0

                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                        if leftEAR < EYE_AR_THRESH:
                            left_COUNTER += 1
                        elif rightEAR > EYE_AR_THRESH:
                            right_COUNTER += 1

                        else:
                            if right_COUNTER >= EYE_AR_CONSEC_FRAMES:
                                right_TOTAL += 1
                                blink_sequence.append('r')
                            elif left_COUNTER >= EYE_AR_CONSEC_FRAMES:
                                left_TOTAL += 1
                                blink_sequence.append('l')

                            right_COUNTER = 0
                            left_COUNTER = 0
                            arr_len = len(blink_sequence)
                            string = blink_pattern[id]
                            lst = []

                            for letter in string:
                                lst.append(letter)

                            print(blink_pattern[id])


                            print(lst)
                            if arr_len > 3:
                                for i in range(arr_len - 2):
                                    if blink_sequence[i] == lst[0] and \
                                            blink_sequence[i + 1] == lst[1] and \
                                            blink_sequence[i + 2] == lst[2]:
                                        s = 'authenticated'
                                        finally_done = True

                            #print(blink_pattern[id])
                            #print(blink_sequence)
                            #print(s)

                            right_COUNTER = 0
                            left_COUNTER = 0

                        cv2.putText(frame, "Right Blinks: {}".format(right_TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "Right EAR: {:.2f}".format(rightEAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "Left Blinks: {}".format(left_TOTAL), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "left EAR: {:.2f}".format(leftEAR), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)

                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF
                    face_is_right = False

                    if finally_done == True:
                        break

                if finally_done == True:
                    from PIL import Image, ImageFont, ImageDraw

                    my_image = Image.open("a6373870819f3f07b8a1e07fa6ac9f45.jpg")
                    title_text = "You are real"
                    image_editable = ImageDraw.Draw(my_image)
                    image_editable.text((15, 15), title_text, (237, 230, 211))
                    my_image.save("result.jpg")
                    im = Image.open("result.jpg")

                    im.show()

                cv2.destroyAllWindows()
                vs.stop()

            else:
                print('no')
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            face_is_right = False

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.putText(img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('camera', img)
    # displaying the frame with fps
    #cv2.imshow('frame', gray)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if is_time == True:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


