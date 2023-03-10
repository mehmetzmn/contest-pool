import cv2
import numpy as np

path = 'Shot-Predictor-Video.mp4'


capture = cv2.VideoCapture(path)
cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

if not capture.isOpened():
    print("Unable to open" + path)
    exit(0)

class callback:
    @staticmethod
    def callback(self):
        pass

def create_trackerbar(window_name='frame', callback=callback.callback):
    # create trackbars for color change
    lowH, highH = 0, 179
    lowS, highS = 0, 255
    lowV, highV = 0, 255
    cv2.createTrackbar('lowH', window_name, lowH, highH, callback)
    cv2.createTrackbar('highH', window_name, highH, highH, callback)

    cv2.createTrackbar('lowS', window_name, lowS, highS, callback)
    cv2.createTrackbar('highS', window_name, highS, highS, callback)

    cv2.createTrackbar('lowV', window_name, lowV, highV, callback)
    cv2.createTrackbar('highV', window_name, highV, highV, callback)

# not creating trackbar for now
# create_trackerbar()


def get_trackbar(frame, window_name='frame'):
    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowH', window_name)
    ihighH = cv2.getTrackbarPos('highH', window_name)
    ilowS = cv2.getTrackbarPos('lowS', window_name)
    ihighS = cv2.getTrackbarPos('highS', window_name)
    ilowV = cv2.getTrackbarPos('lowV', window_name)
    ihighV = cv2.getTrackbarPos('highV', window_name)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    return frame

# NOTE: lowV: 44, highH: 69, lowS: 0, highS: 158, lowV: 133, highV: 255 --> white ball

lower_range = (44, 0, 133)
upper_range = (69, 158, 255)

while True:
    _, frame = capture.read()
    
    width = 640
    height = 480
    dim = (width, height)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    if frame is None:
        break

    # not calling not existing trackbar    
    # hsv_trackbar = get_trackbar(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10:  # Adjust this value according to the size of the ball in the video
            x, y, w, h = cv2.boundingRect(cnt)
            print(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # cv2.imshow('hsv', hsv)
    cv2.imshow('frame', frame)

    keyboard = cv2.waitKey(30)
    if keyboard == 27 or keyboard == ord('q'):
        break
