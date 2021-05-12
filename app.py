import cv2
import numpy as np
import mediapipe as mp
from numpy.lib.type_check import imag
import pyautogui as pg

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
)
mpDraw = mp.solutions.drawing_utils


def resizeimage(img):
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resizedImage = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resizedImage


def findDistanceOfIndividualFingers(points, cors):
    first = cors[points[1]]
    sec = cors[0]
    distance = int(
        (((sec[1]-first[1])**2)+((sec[0]-first[0])**2))**(0.5))
    return distance


def finger_inside_box(finger, start_point, endpoint):
    return finger[0] > start_point[0] and finger[0] < end_point[0]


fingertips = {"thumb": [2, 4], "index": [6, 8], "middle": [
    10, 11], "ring": [14, 16], "pinky": [18, 20], }
width, height = pg.size()
white_image = (np.zeros([height, width, 3], dtype=np.uint8))
white_image.fill(255)
start_point = (20, 30)
pg.FAILSAFE = False
end_point = (200, 180)
cap = cv2.VideoCapture(0)
finger_function = "move"

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resizedImage = resizeimage(image)
    imgRgb = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRgb)

    position_scale = 100
    white_image[height-resizedImage.shape[0]-position_scale:height-position_scale,
                width-resizedImage.shape[1]-position_scale:width-position_scale, 0:] = resizedImage
    green = ((width-resizedImage.shape[1],
              70,)), ((width-resizedImage.shape[1]+100,
                       140,))
    blue = ((green[0][0], green[0][1]+green[0][1]+10,),
            (green[1][0], green[1][1]+green[1][1]-50,))
    red = ((blue[0][0], blue[0][1]+blue[0][1]-30,),
           (blue[1][0], blue[1][1]+blue[1][1]-100,))

    if results.multi_hand_landmarks:
        hands_list = []
        for handLndms in (results.multi_hand_landmarks):
            hand = {}
            for id, lm in enumerate(handLndms.landmark):
                h, w, c = resizedImage.shape
                hx, hy = int(lm.x*w), int(lm.y*h)
                hand[id] = (hx, hy)
            hands_list.append(hand)
            mpDraw.draw_landmarks(resizedImage, handLndms,
                                  mpHands.HAND_CONNECTIONS)
            indexfinger = hand[8]
            cv2.circle(resizedImage, indexfinger, 5, (255, 0, 0), -1)
        x_value = int(np.interp(indexfinger[0], [
            start_point[0], end_point[0]], [0, width]))
        y_value = int(np.interp(indexfinger[1], [
            start_point[1], end_point[1]], [0, height]))
        if(finger_inside_box(indexfinger, start_point, end_point)):

            # print(f" mouse values {(x_value, y_value)}")
            # print(f" box values {(blue[0], blue[1])}")
            # print(x_value,blue[0][0])
            if(x_value > blue[0][0] and x_value < blue[1][0]):
                finger_function = "blue"
                pg.moveTo(x_value, y_value)

            # print(finger_inside_box((x_value, y_value), blue[0], blue[1]))
            if finger_function == "move":
                pg.moveTo(x_value, y_value)
    cv2.putText(white_image, "Colors", (width-resizedImage.shape[1],
                                        40,),
                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 2)
    cv2.rectangle(white_image, green[0], green[1], (255, 255, 0), -1)
    cv2.rectangle(white_image, blue[0], blue[1], (255, 0, 0), -1)
    cv2.rectangle(white_image, red[0], red[1], (0, 0, 255), -1)
    cv2.line(white_image, (width-resizedImage.shape[1]-120,
                           0,), (width-resizedImage.shape[1]-120, height,), (255, 100, 0), 1)
    cv2.rectangle(resizedImage, start_point, end_point, (255, 255, 0), 2)

    cv2.imshow("white image", white_image)

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
