

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

while face_is_right:
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

                print(lst)
                if arr_len > 3:
                    for i in range(arr_len - 2):
                        if blink_sequence[i] == lst[0] and \
                                blink_sequence[i + 1] == lst[1] and \
                                blink_sequence[i + 2] == lst[2]:
                            s = 'authenticated'
                            finally_done = True

                print(blink_pattern[id])
                print(blink_sequence)
                print(s)

                right_COUNTER = 0
                left_COUNTER = 0

            cv2.putText(frame, "Right Blinks: {}".format(right_TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Right EAR: {:.2f}".format(rightEAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Left Blinks: {}".format(left_TOTAL), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "left EAR: {:.2f}".format(leftEAR), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        face_is_right = False



        if finally_done ==True:
            break

    if finally_done ==  True:
        from PIL import ImageFont
        import tkinter as tk
        from PIL import Image, ImageTk
        from itertools import count, cycle


        class ImageLabel(tk.Label):
            """
            A Label that displays images, and plays them if they are gifs
            :im: A PIL Image instance or a string filename
            """

            def load(self, im):
                if isinstance(im, str):
                    im = Image.open(im)
                frames = []

                try:
                    for i in count(1):
                        frames.append(ImageTk.PhotoImage(im.copy()))
                        im.seek(i)
                except EOFError:
                    pass
                self.frames = cycle(frames)

                try:
                    self.delay = im.info['duration']
                except:
                    self.delay = 100

                if len(frames) == 1:
                    self.config(image=next(self.frames))
                else:
                    self.next_frame()

            def unload(self):
                self.config(image=None)
                self.frames = None

            def next_frame(self):
                if self.frames:
                    self.config(image=next(self.frames))
                    self.after(self.delay, self.next_frame)

            from PIL import Image, ImageDraw, ImageSequence
            import io

            im = Image.open('Smiling Leo Perfect GIF.gif')

            # A list of the frames to be outputted
            frames = []
            # Loop over each frame in the animated image
            for frame in ImageSequence.Iterator(im):
                # Draw the text on the frame
                draw = ImageDraw.Draw(im)
                txt = f"YES!! You are authenticated {id}"
                font = ImageFont.truetype("arial.ttf", 35)
                draw.text((10, 350), txt, font=font)  # put the text on the image  # save it
                # d.text((250, 350), f"YES!! you are someone",font_size)
                # del d

                # However, 'frame' is still the animated image with many frames
                # It has simply been seeked to a later frame
                # For our list of frames, we only want the current frame

                # Saving the image without 'save_all' will turn it into a single frame image, and we can then re-open it
                # To be efficient, we will save it to a stream, rather than to file
                b = io.BytesIO()
                frame.save(b, format="GIF")
                frame = Image.open(b)

                # Then append the single frame image to a list of frames
                frames.append(frame)
            # Save the frames as a new image
            frames[0].save('out.gif', save_all=True, append_images=frames[1:])


        # demo :
        root = tk.Tk()
        lbl = ImageLabel(root)
        lbl.pack()
        lbl.load('out.gif')
        root.mainloop()

    cv2.destroyAllWindows()
    vs.stop()


