import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from tensorflow import keras
from matplotlib import pyplot as plt

model = keras.models.load_model('./model')

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    result = hands.process(frame)
    hand_landmarks = result.multi_hand_landmarks

    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
    if hand_landmarks:
        for handLMs in hand_landmarks:
            for lm in handLMs.landmark:
                #mp_drawing.draw_landmarks(frame, handLMs, mp_hands.HAND_CONNECTIONS)
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(frame, (x_min - 50, y_min - 50), (x_max + 50, y_max + 50), (0, 255, 0), 2)

    if x_min > 50 and x_max != 0 and y_min > 50 and y_max != 0:
        x_max = x_min + (y_max - y_min)
        R = frame[y_min - 50 : y_max + 50, x_min - 50 : x_max + 50, 0]
        G = frame[y_min - 50 : y_max + 50, x_min - 50 : x_max + 50, 1]
        B = frame[y_min - 50 : y_max + 50, x_min - 50 : x_max + 50, 2]
        gray_frame = 0.2989 * R + 0.587 * G + 0.114 * B
        np_gray_frame = np.array(gray_frame)
        image = Image.fromarray(np_gray_frame)
        resize_image = image.resize((28, 28))
        #plt.imshow(resize_image)
        #plt.show()
        ar = np.array(resize_image)
        prediction = model.predict([np.array(resize_image).tolist()])
        pred_char = chr(np.argmax(prediction) + 65)
        cv2.putText(frame, 'Letter: ' + pred_char, (x_min, y_min - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

#result = model.predict(test_input)
#print(result)
