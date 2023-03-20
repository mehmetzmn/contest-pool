import cv2
import numpy as np


# np.set_printoptions(threshold=sys.maxsize)
path = 'Shot-Predictor-Video.mp4'


capture = cv2.VideoCapture(path)
cv2.namedWindow('hsv_trackbar', cv2.WINDOW_AUTOSIZE)

if not capture.isOpened():
    print("Unable to open" + path)
    exit(0)

def on_trackbar(val):
    pass

def create_trackerbar(window_name='frame', track=on_trackbar):

    # create trackbars for color change
    lowH, highH = 0, 179
    lowS, highS = 0, 255
    lowV, highV = 0, 255
    
    cv2.createTrackbar('lowH', window_name, lowH, highH, track)
    cv2.createTrackbar('highH', window_name, highH, highH, track)

    cv2.createTrackbar('lowS', window_name, lowS, highS, track)
    cv2.createTrackbar('highS', window_name, highS, highS, track)

    cv2.createTrackbar('lowV', window_name, lowV, highV, track)
    cv2.createTrackbar('highV', window_name, highV, highV, track)

# NOTE: not creating trackbar for now
create_trackerbar(window_name='hsv_trackbar')


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

    hsv_values = {'ilowH': ilowH, 'ihighH': ihighH, 'ilowS': ilowS,
                  'ihighS': ihighS, 'ilowV': ilowV, 'ihighV': ihighV}

    return frame, hsv_values

# NOTE: lowV: 44, highH: 69, lowS: 0, highS: 158, lowV: 133, highV: 255 --> white ball

white_ball_lower_range = (40, 0, 98)
white_ball_upper_range = (79, 87, 255)

stick_lower_range = (21, 3, 154)
stick_upper_range = (89, 48, 195)

while True:
    success, frame = capture.read()
    # frame = cv2.imread("normal.png")
    # frame = cv2.imread('new_image.png')
    # frame = frame[10: 400, 51:588] # for image
    if frame is None: break

    width = 1280
    height = 720
    dim = (width, height)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    # frame = frame[10: 400, 51:588] # for video

    # NOTE: not calling not existing trackbar    
    hsv_trackbar, hsv_values = get_trackbar(frame, window_name='hsv_trackbar')
    print(hsv_values)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, white_ball_lower_range, white_ball_upper_range)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10:  # Adjust this value according to the size of the ball in the video
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

    # cv2.imshow('hsv', hsv)
    cv2.imshow('hsv_trackbar', hsv_trackbar)
    cv2.imshow('frame', frame)
    

    keyboard = cv2.waitKey(30)
    if keyboard == 27 or keyboard == ord('q'):
        break

# {'ilowH': 21, 'ihighH': 89, 'ilowS': 3, 'ihighS': 48, 'ilowV': 154, 'ihighV': 195
# {'ilowH': 64, 'ihighH': 99, 'ilowS': 20, 'ihighS': 102, 'ilowV': 145, 'ihighV': 212}
# {'ilowH': 41, 'ihighH': 76, 'ilowS': 0, 'ihighS': 111, 'ilowV': 109, 'ihighV': 255}