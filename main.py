import cv2
import numpy as np
from ultralytics import YOLO
import natsort
from tqdm import tqdm
import os


def dotted_line(image, start_point, end_point, direction_vector, color=(0, 255, 0), thickness=3, line_type=cv2.LINE_AA, dash_length=10, gap_length=5):

    # Draw the dotted line by iterating over segments
    while True:
        cv2.line(image, start_point, end_point, color, thickness, line_type)

        start_point = (end_point[0] + direction_vector[0] * gap_length,
                       end_point[1] + direction_vector[1] * gap_length)
        end_point = (start_point[0] + direction_vector[0] * dash_length,
                     start_point[1] + direction_vector[1] * dash_length)

        if all(isinstance(i, int) for i in [start_point[0], start_point[1], end_point[0], end_point[1]]):
            print("start_point", start_point)
            print("end_point", end_point)
        if start_point[0] >= image.shape[1] or start_point[1] >= image.shape[0]:
            break

    # Display the image
    cv2.imshow('Dotted Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Create a function that measure the distande between cue stick and balls,
# the closest ball to the cue stick is the target ball


def distance_between_objects(obj1_coordinates, obj2_coordinates):
    x1_1, y1_1, x1_2, y1_2 = obj1_coordinates
    x2_1, y2_1, x2_2, y2_2 = obj2_coordinates

    center_point_obj1 = ((x1_1 + x1_2) / 2, (y1_1 + y1_2) / 2)
    center_point_obj2 = ((x2_1 + x2_2) / 2, (y2_1 + y2_2) / 2)

    distance = np.sqrt((center_point_obj1[0] - center_point_obj2[0]) ** 2 + (
        center_point_obj1[1] - center_point_obj2[1]) ** 2)

    return distance


tmp = {0: [[299.4883, 315.7211, 334.8784, 348.0699], 164.2721],
       1: [[246.6917, 196.2286, 283.2608, 229.7630], 292.9715]}

# A function that returns highest value from this dictionary's key values second element


def return_highest_value_from_dict(dct):
    temp = []
    for key, value in dct.items():
        temp.append(value[1])
        max_val = max(temp)
        if len(temp) != 1 and max_val == value[1]:
            return key, value[0]


# direction_vector = (start_point[0] - end_point[0],
#                     start_point[1] - end_point[1])

# end_point_line = (
#     start_point[0] + direction_vector[0], start_point[1] + direction_vector[1])
# print(end_point_line)

# cv2.line(frame, (start_point[0]+3, start_point[1]),
#             (end_point_line[0]-3, end_point_line[1]), (0, 0, 255), 3)

# cv2.circle(
#     frame, (end_point_line[0]+3, end_point_line[1]), 10, (0, 0, 255), 5)
# NOTE: Above turned into a function below
def line_tip_of_cue_stick(start_point, end_point, image, pixel_sliding=3, line_color=(0, 0, 255), line_thickness=3, circle_color=(0, 0, 255), circle_radius=10, circle_thickness=5):
    direction_vector = (
        start_point[0] - end_point[0], start_point[1] - end_point[1])

    end_point_line = (start_point[0] + direction_vector[0],
                      start_point[1] + direction_vector[1])
    print(end_point_line)

    cv2.line(image, (start_point[0] + pixel_sliding, start_point[1]),
             (end_point_line[0] - pixel_sliding, end_point_line[1]), line_color, line_thickness)
    cv2.circle(image, (end_point_line[0] + pixel_sliding,
               end_point_line[1]), circle_radius, circle_color, circle_thickness)


def find_location_of_cue_stick():
    # measure between center of the roi of cue stick and horizantal line of the frame
    # check sign of the subtraction, in this case, both x and y axis signs give away the position of the cue stick

    pass


def find_coordinates(target_ball_center, start_point_of_cue_stick, distance):
    # find the angle from the center of the target ball to the start point of the cue stick
    # angle equation: theta = arctan((y2-y1)/(x2-x1))
    # from there with these equations --> x2 = x1 + d * cos(theta)
    #                                     y2 = y1 + d * sin(theta)
    # we can find the coordinates of the point where the cue stick should hit the target ball

    theta = np.arctan((target_ball_center[1] - start_point_of_cue_stick[1]) /
                      (target_ball_center[0] - start_point_of_cue_stick[0]))

    x2 = target_ball_center[0] + distance * np.cos(theta)
    y2 = target_ball_center[1] + distance * np.sin(theta)

    return x2, y2


# x, y = find_coordinates((264.5, 212.5), (390, 617), 10)

# print(x, y)


def find_mean_of_coordinates(point_list, is_max=False):
    temp_list = sorted(point_list, key=lambda coord: (
        coord[0]**2 + coord[1]**2), reverse=is_max)[:10]
    x = []
    y = []
    for coordinate in temp_list:
        x.append(coordinate[0])
        y.append(coordinate[1])

    result = np.array([np.mean(x), np.mean(y)]).astype(np.int32)
    return result


ROOT_PATH = os.getcwd()

PATH = f"{ROOT_PATH}/frames"

MODEL_PATH_SEGMENT = f"{ROOT_PATH}/runs/segment/train/weights/best.pt"


def detect(PATH, MODEL_PATH):

    IMAGES = []
    for img in os.listdir(PATH):
        IMAGES.append(os.path.join(PATH, img))

    IMAGES = natsort.natsorted(IMAGES)
    model = YOLO(MODEL_PATH)
    model.to('mps')

    IMAGES = IMAGES[400:1000]

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
        results = model.predict(frame, show=False, retina_masks=True,
                                show_labels=True, boxes=False, show_conf=True, save_txt=False, save=False)
        print(type(results), len(results))
        print(type(results[0]), len(results[0]))

        H, W = frame.shape[0], frame.shape[1]

        from yolo_segmentation import YOLOSegmentation

        model_segment = YOLOSegmentation(MODEL_PATH_SEGMENT)

        _, class_ids, segment_contours, _ = model_segment.detect(frame)

        for class_id, segment_contour in zip(class_ids, segment_contours):
            if class_id == 1:
                cv2.polylines(frame, [segment_contour], True, (0, 255, 0), 2)

                print(max(segment_contour, key=lambda x: x[0]))

                cv2.circle(frame, (100, 612), 5, (0, 255, 0), -1)
                cv2.circle(frame, (100, 10), 5, (0, 255, 0), -1)
                cv2.circle(frame, (1180, 10), 5, (0, 255, 0), -1)

                # fit a line to the points
                vx, vy, x, y = cv2.fitLine(
                    segment_contour, cv2.DIST_L2, 0, 0.01, 0.01)

                print("vx", vx, "vy", vy, "x", x, "y", y)

                # calculate the endpoints of the line
                lefty = int((-x*vy/vx) + y)  # y = mx + c
                righty = int(((frame.shape[1]-x)*vy/vx)+y)
                print("lefty", lefty, "righty", righty)

                pt1 = (frame.shape[1]-1, righty)
                pt2 = (0, lefty)

                # draw the line on the image
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
                cv2.circle(frame, (pt2[0], pt2[1]), 10, (255, 0, 255), -1)
                cv2.putText(
                    frame, "start point", (pt1[0], pt1[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(
                    frame, "end point", (pt2[0], pt2[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 255), -1)

            else:
                continue
                # print(segment_contour)

        ##################
        # NOTE: This is part of different attributes of the mask object
        # results[0].masks.data --> type is class 'torch.Tensor', len is 9, returns lists of tensors [0., 1.]
        # results[0].masks.xyn --> type is class 'list', len is 9, returns lists of segments
        # results[0].masks.segments --> type is class 'list', len is 9, returns lists of segments ## This raises "masks.segments is deprecated, use masks.xyn instead"
        # results[0].masks.masks --> type is class 'torch.Tensor', len is 9, returns lists of tensors [0., 1.]
        ##################

        cv2.imshow("Frame", frame)

        keyboard = cv2.waitKey(1)
        if keyboard == 27 or keyboard == ord('q'):
            break


detect(PATH, MODEL_PATH_SEGMENT)
