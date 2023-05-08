from ultralytics import YOLO
from tqdm import tqdm
import natsort
import os
import cv2
import numpy as np

from utils import project_utils

ROOT_PATH = os.getcwd()

PATH = f"{ROOT_PATH}/frames"

# for segment
MODEL_PATH_SEGMENT = f"{ROOT_PATH}/runs/segment/train/weights/best.pt"

# for detect
MODEL_PATH_DETECT = f"{ROOT_PATH}/runs/detect/train/weights/best.pt"

# yaml file path
PATH_YAML = f"{ROOT_PATH}/Billiard objects detection.v3i.yolov8/data.yaml"


def detect(PATH, MODEL_PATH):

    IMAGES = []
    for img in os.listdir(PATH):
        IMAGES.append(os.path.join(PATH, img))

    IMAGES = natsort.natsorted(IMAGES)
    model = YOLO(MODEL_PATH)
    model.to('mps')

    # tespit edilemeyen topları debug et
    # IMAGES = IMAGES[575:650]
    # IMAGES = IMAGES[1010:1100]
    # IMAGES = IMAGES[70:80]

    mask_list = {}
    initial_position = []
    for index in tqdm(range(len(IMAGES))):

        print("Current image:", IMAGES[index], "\n")

        frame = cv2.imread(IMAGES[index])

        width = 1280
        height = 720
        dim = (width, height)

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(mask=False, probs=False, boxes=False)

        boxes = results[0].boxes

        ball_num = 0
        distances = {}

        cue_stick_coordinates = None
        yolo_object_dict = {}

        for obj_xy, obj_wh, clas in zip(boxes.xyxy, boxes.xywh, boxes.cls):
            object_xy = []
            object_wh = []

            object_xy.append(obj_xy)
            object_wh.append(obj_wh)

            yolo_object_dict[clas] = [object_xy, object_wh]

        # Extract the region of interest (ROI) using the bounding box coordinates
        for key in sorted(yolo_object_dict.keys(), reverse=True):
            # NOTE: Class 1 is the cue stick, avoid other classes
            if key == 1:
                print("Cue sick detected")
                x1 = int(yolo_object_dict[key][0][0][0])
                y1 = int(yolo_object_dict[key][0][0][1])
                x2 = int(yolo_object_dict[key][0][0][2])
                y2 = int(yolo_object_dict[key][0][0][3])
                roi = frame[y1:y2, x1:x2]
                # cue stick coordinates
                cue_stick_coordinates = yolo_object_dict[key][0][0]

                start_point = (x1, y1)
                end_point = (x2, y2)

                cue_stick_direction, cue_stick_center = project_utils.utils.get_cue_direction(
                    cue_stick_coordinates)
                print("DIRECTION OF CUE STICK", cue_stick_direction)
                cv2.circle(frame, (x1 + 2, y1), 5, (0, 255, 255), -1)
                # cv2.circle(frame, (x2, y2), 5, (0, 255, 255), -1)

                project_utils.utils.line_tip_of_cue_stick(
                    start_point, end_point, cue_stick_direction, frame)
                # print("Inside of if statement", cue_stick_coordinates)
            else:
                # NOTE: Class 1 is the cue stick, avoid drawing circles around it
                project_utils.utils.draw_cirles(
                    yolo_object_dict[key][1][0], frame, key)

            # print("Outside of if statement", cue_stick_coordinates)
            # print("class is ", key)
            if cue_stick_coordinates is not None:
                ball_variables = []
                temp = []
                if key == 0:
                    ball_coordinates = yolo_object_dict[key][0][0]
                    # print("ball coordinates", ball_coordinates)

                    distance = project_utils.utils.distance_between_objects(
                        ball_coordinates, cue_stick_coordinates)

                    temp.append(distance)
                    ball_variables.append(ball_coordinates)
                    ball_variables.append(temp)
                    distances[ball_num] = ball_variables
                    ball_num += 1

            else:
                print("cue stick coordinates is None")

            # print("distances -->", distances)

            if len(distances) == 0:
                print("distances is empty")
            # for comparison distances length should be 2
            elif len(distances) == 2:
                print("DISTANCES -->", distances)

                # return the highest distance value and its key, which is the target ball
                results = project_utils.utils.return_highest_value_from_dict(
                    distances)

                # Extract the target ball coordinates
                target_ball, target_ball_coordinate = results if results else (0, [
                    0, 0])

                # Extract the white ball coordinates
                for key, value in distances.items():
                    if key != target_ball:
                        white_ball, white_ball_coordinate = key, value
                        print("white ball -->", key, "value -->", value)

                # Mark the target ball on the frame as "target"
                cv2.putText(frame, "target", (
                    int(target_ball_coordinate[0]), int(target_ball_coordinate[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                ################

                # get the center point of the target ball, use it to get theta angle
                center_point_target_ball = ((int(target_ball_coordinate[0]) + int(
                    target_ball_coordinate[2])) / 2, (int(target_ball_coordinate[1]) + int(target_ball_coordinate[3])) / 2)

                center_point_white_ball = ((int(white_ball_coordinate[0][0]) + int(
                    white_ball_coordinate[0][2])) / 2, (int(white_ball_coordinate[0][1]) + int(white_ball_coordinate[0][3])) / 2)

                # Initial position of the target ball
                initial_position.append(
                    [int(target_ball_coordinate[0]), int(target_ball_coordinate[1])])

                # To draw hit line check if the target ball has moved with initial position
                if len(initial_position) >= 2 and initial_position[0][0] - initial_position[-1][0] < 10:

                    new_x, new_y = project_utils.utils.find_coordinates(
                        center_point_target_ball, center_point_white_ball, (x2, y2), 40)

                    theta = np.arctan((y2 - y1) / (x2 - x1))
                    # aci 90 - theta
                    new_x2 = center_point_target_ball[0] + \
                        32 * np.cos(90 - theta)
                    new_y2 = center_point_target_ball[1] + \
                        32 * np.sin(90 - theta)
                    cv2.circle(frame, (int(new_x2), int(new_y2)),
                               16, (255, 255, 0), -1)
                    cv2.line(frame, (int(x1+2), int(y1)), (int(new_x2), int(new_y2)),
                             (255, 255, 0), 4)

        # print("yolo_object_dict -->", yolo_object_dict)

            # NOTE: Displaying the ROI is optional, currently disabled
            # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # # Find contours in the grayscale ROI
            # contours, hierarchy = cv2.findContours(
            #     gray_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # Example contour filtering based on contour area
            # min_area = 100  # minimum contour area threshold
            # filtered_contours = [
            #     contour for contour in contours if cv2.contourArea(contour) > min_area]

            # # Draw contours on the ROI
            # cv2.drawContours(roi, filtered_contours, -1, (0, 255, 0), 2)

        # NOTE: Uncomment the following lines to save the annotated frame mask.xy values to a file
        # if os.stat("maskValues.txt").st_size == 0:
        #     mask_list.update({f"Current image: {IMAGES[index]}": mask.xy})
        #     f = open("maskValues.txt", "w")
        #     f.write(str(mask_list))
        #     f.close()
        # else:
        #     print("File is not empty")

        # Display the annotated frame
        # cv2.imshow("detection window", annotated_frame)

        cv2.imshow("contours", frame)
        keyboard = cv2.waitKey(30)
        if keyboard == 27 or keyboard == ord('q'):
            break


# makedir("segmented_frames")
detect(PATH, MODEL_PATH_DETECT)
