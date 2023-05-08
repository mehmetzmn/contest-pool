import json
import skvideo.datasets
import cv2
import numpy as np
import os
import skvideo.io


print("current directory is : " + os.getcwd())


path = 'Shot-Predictor-Video.mp4'
path = '/Users/archosan/Desktop/Python projects/contest/contest-pool/tfod/video.mp4'
pathOut = 'frames'
path_yolo = "video-yolo.mp4"


def read_video(path):
    capture = cv2.VideoCapture(path)
    cv2.namedWindow('hsv_trackbar', cv2.WINDOW_AUTOSIZE)

    if not capture.isOpened():
        print("Unable to open" + path)
        exit(0)

    frame_count = 0

    while True:
        success, frame = capture.read()

        if frame is None:
            break

        width = 1280
        height = 720
        dim = (width, height)

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        frame_count += 1
        # cv2.imwrite(pathOut + "/frame%d.jpg" % frame_count, frame)

        cv2.imshow('window', frame)

        keyboard = cv2.waitKey(30)
        if keyboard == 27 or keyboard == ord('q'):
            break

    print("Number of total frames:", frame_count)


def frame_to_vid(num_frames, width, height, ch, image_dir, video_name):

    ROOT_DIR = os.getcwd()

    if video_name not in os.listdir(ROOT_DIR):
        print("Video does not exist")

        out_video = np.empty(
            [num_frames, height, width, ch], dtype=np.uint8)

        out_video = out_video.astype(np.uint8)

        for index, image in enumerate(os.listdir(image_dir)):

            img = cv2.imread(f"{image_dir}/frame{index + 1}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out_video[index] = img

        # Writes the the output image sequences in a video file
        skvideo.io.vwrite(video_name, out_video)


frame_to_vid(1222, 1280, 720, 3,
             '/Users/archosan/Desktop/Python projects/contest/contest-pool/detected_frames', 'video-yolo.mp4')

# read_video(path_yolo)

frame_to_vid(1222, 1280, 720, 3,
             "/Users/archosan/Desktop/Python projects/contest/contest-pool/segmented_frames", "video-yolo-segmented.mp4")
