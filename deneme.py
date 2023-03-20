import cv2
import numpy as np

# frame = cv2.imread('normal.png')
# frame = cv2.imread('new_image.png')
frame = cv2.imread('rescaled_img.png')

# frame2 = frame[10: 400, 51:588]

white_ball_lower_range = (41, 0, 109)
white_ball_upper_range = (76, 111, 255)

# stick_lower_range = (64, 20, 145)
# stick_upper_range = (99, 102, 212)


# all_values = {'white_ball': (white_ball_lower_range, white_ball_upper_range),
#               'pool_stick': (stick_lower_range, stick_upper_range)}




# for hsv_key, hsv_value in all_hsv_values.items():
#     print(hsv_key)
#     print(hsv_value[0], "--> lower range")
#     print(hsv_value[1], "--> upper range")

img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 100, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
cv2.imshow('frame', frame)

# img = cv2.medianBlur(img_gray, 5)
# ret, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#             cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#             cv2.THRESH_BINARY, 11, 2)


# cv2.imshow('Original Image', img)
# cv2.imshow('Global Thresholding (v = 127)', th1)
# cv2.imshow('Adaptive Mean Thresholding', th2)
# cv2.imshow('Adaptive Gaussian Thresholding', th3)

# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# # mask = cv2.inRange(hsv, hsv_value[0], hsv_value[1])
# mask = cv2.inRange(hsv, white_ball_lower_range, white_ball_upper_range)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
# # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# all_detected_areas = sorted(set([cv2.contourArea(cnt) for cnt in contours]), reverse=True)
# print(all_detected_areas)
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if 277 < area < 815:  # Adjust this value according to the size of the ball in the video
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     elif 815 < area < 1410:
#         cv2.drawContours(frame, cnt, -1, (0, 0, 255), 3)

# cv2.imshow('frame', frame)
# cv2.imshow('masked_image', mask)
# cv2.imshow('img_gray', img_gray)
keyboard = cv2.waitKey(0)
if keyboard == 27 or keyboard == ord('q'):
    cv2.destroyAllWindows()