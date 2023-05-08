import os
import numpy as np
import cv2

from ultralytics import YOLO


class utils:

    @staticmethod
    def makedir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def save_images(path, image, annotated_frame):
        OUTPUT_ROOT = path
        OUTPUT_PATH = os.path.join(OUTPUT_ROOT, image[68:])
        print(OUTPUT_PATH)
        cv2.imwrite(OUTPUT_PATH, annotated_frame)

    @staticmethod
    def train(PATH_YAML, train_type):
        # load a pretrained model (recommended for training)
        if train_type == "segment":
            model = YOLO('yolov8x-seg.pt')
            model.to('mps')
        else:
            model = YOLO('yolov8x.pt')
            model.to('mps')

        # Use the model
        model.train(data=PATH_YAML,  # path to training data
                    epochs=100)  # train the model
        metrics = model.val()  # evaluate model performance on the validation set

    @staticmethod
    def show_mask(W, H, results):
        for result in results:
            for j, mask in enumerate(result.masks.segments):
                mask = mask.numpy() * 255

                cv2.resize(mask, (W, H))
                cv2.imshow("output_mask", mask)

    @staticmethod
    def draw_cirles(obj, frame, clas):
        x, y, w, h = obj  # Extract bounding box coordinates
        radius = int(h/2)  # Set circle radius
        color = (0, 255, 0)  # Set circle color (green)
        thickness = 2  # Set circle thickness

        # if class is 0 (ball), draw more wide circle
        if clas == 0:
            # Draw circle around object center
            print("radius", radius)
            cv2.circle(frame, (int(x), int(y)), radius + 1, color, thickness)
        else:
            # Draw circle around object center
            cv2.circle(frame, (int(x), int(y)), radius, color, thickness)

        # Display frame with circles around detected objects
        # cv2.imshow('Detected Objects', frame)

    @staticmethod
    def distance_between_objects(obj1_coordinates, obj2_coordinates):
        x1_1, y1_1, x1_2, y1_2 = obj1_coordinates
        x2_1, y2_1, x2_2, y2_2 = obj2_coordinates

        center_point_obj1 = ((x1_1 + x1_2) / 2, (y1_1 + y1_2) / 2)
        center_point_obj2 = ((x2_1 + x2_2) / 2, (y2_1 + y2_2) / 2)

        distance = np.sqrt((center_point_obj1[0] - center_point_obj2[0]) ** 2 + (
            center_point_obj1[1] - center_point_obj2[1]) ** 2)

        return distance

    @staticmethod
    def return_highest_value_from_dict(dct):
        # temp = []
        # for key, value in dct.items():
        #     temp.append(value[1])
        #     max_val = max(temp)
        #     print("MAX_VAL", max_val, "\n", "VALUE", value[1])
        #     if len(temp) != 1 and max_val == value[1]:
        #         print("key, value pair", key, value[0])
        #         return key, value[0]
        max_val, key, value_0 = None, None, None

        for k, v in dct.items():
            if max_val is None or v[1] > max_val:
                max_val = v[1]
                key = k
                value_0 = v[0]
            else:
                continue

        return key, value_0

    @staticmethod
    def line_tip_of_cue_stick(start_point, end_point, direction_vector, image, pixel_sliding=3, line_color=(0, 0, 255), line_thickness=3, circle_color=(0, 0, 255), circle_radius=10, circle_thickness=5):
        # direction_vector = (
        #     start_point[0] - end_point[0], start_point[1] - end_point[1])

        # print(type(list(direction_vector)[0]))

        end_point_line = (start_point[0] + int(list(direction_vector)[0]),
                          start_point[1] + int(list(direction_vector)[1]))
        # print(end_point_line)

        # cv2.line(image, (start_point[0] + pixel_sliding, start_point[1]),
        #  (end_point_line[0] - pixel_sliding, end_point_line[1]), line_color, line_thickness)
        # cv2.circle(image, (end_point_line[0] + pixel_sliding,
        #                    end_point_line[1]), circle_radius, circle_color, circle_thickness)

    @staticmethod
    def get_cue_direction(cue_stick_coords):
        # Compute the center point of the cue stick
        cue_stick_center = np.array([(cue_stick_coords[0] + cue_stick_coords[2])/2,
                                    (cue_stick_coords[1] + cue_stick_coords[3])/2])

        # Compute the direction vector
        cue_stick_tip = np.array([cue_stick_coords[2], cue_stick_coords[3]])
        cue_direction = cue_stick_tip - cue_stick_center
        # normalize to unit length
        cue_direction /= np.linalg.norm(cue_direction)

        return cue_direction, cue_stick_center

    @staticmethod
    def find_coordinates(target_ball_center, white_ball_center, start_point_of_cue_stick, distance):
        # find the angle from the center of the target ball to the start point of the cue stick
        # angle equation: theta = arctan((y2-y1)/(x2-x1))
        # from there with these equations --> x2 = x1 + d * cos(theta)
        #                                     y2 = y1 + d * sin(theta)
        # we can find the coordinates of the point where the cue stick should hit the target ball

        theta = np.arctan((white_ball_center[1] - start_point_of_cue_stick[1]) /
                          (white_ball_center[0] - start_point_of_cue_stick[0]))

        x2 = target_ball_center[0] + distance * np.cos(theta)
        y2 = target_ball_center[1] + distance * np.sin(theta)

        return x2, y2
