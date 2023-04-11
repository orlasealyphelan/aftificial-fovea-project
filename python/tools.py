import json
import cv2
import csv
import os
import numpy as np


# function to format json file containing annotations from CVAT
# returns two dictionaries: gt_boxes contains only bounding boxes, gt_results contains bounding boxes and class label
# gt_boxes = dictionary with key: Image ID (0-900)
# values: list of bounding boxes in form [x1, y1, x2, y2]
# gt_results = dictionary with key: Image ID (0-900)
# values: list of prediction and bounding boxes in form [class_id, x1, y1, x2, y2]
# class ids = {1: 'car', 2: 'person', 3: 'bicycle'}
def format_json(fname):
    gt_boxes = {}
    gt_results = {}
    f = open(fname)
    data = json.load(f)
    for i in data["annotations"]:
        class_id = i["category_id"]
        x = i["bbox"][0]
        y = i["bbox"][1]
        w = i["bbox"][2]
        h = i["bbox"][3]
        if i["image_id"] in gt_boxes:
            gt_boxes[i["image_id"]].append([x, y, x + w, y + h])
        else:
            gt_boxes[i["image_id"]] = []
            gt_boxes[i["image_id"]].append([x, y, x + w, y + h])
        if i["image_id"] in gt_results:
            gt_results[i["image_id"]].append([class_id, x, y, x + w, y + h])
        else:
            gt_results[i["image_id"]] = []
            gt_results[i["image_id"]].append([class_id, x, y, x + w, y + h])
    return gt_boxes, gt_results


# function to display/save projection results
def disp_side_by_side(rgb_img, event_img, current_frame, folder=None):
    # combine event and rgb images
    concat = np.concatenate((rgb_img, event_img), axis=1)
    # display results or save to file
    if folder is None:
        cv2.imshow('frame0', concat)
        cv2.waitKey(0)
    else:
        if not os.path.isdir(folder):
            os.mkdir(folder)
        name = os.path.join(folder, 'frame' + str(current_frame) + '.jpg')
        cv2.imwrite(name, concat)


# function to draw bounding boxes on image
# boxes = list of bounding box coordinates
# format: [[x1, y1, x2, y2], ... ,[x1, y1, x2, y2]]
# returns annotated image
def draw_rectangles(image, box_list, color, thickness=1):
    for box in box_list:
        start_point = (round(box[0]), round(box[1]))
        end_point = (round(box[2]), round(box[3]))
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


# function to draw bounding boxes on image with class label
# box_list = list of results in format: [[frame, class id, confidence, x1, y1, x2, y2], ...]
# default OD=False selects classification results with classes[2] = rider
# set to True for YOLO labelling where classes[2] = bicycle
# returns annotated image
def draw_rectangles_labelled(image, box_list, color, thickness=2, OD=False):
    classes = {0: 'car: ', 1: 'person: ', 2: 'rider: '}
    for box in box_list:
        start_point = (round(box[3]), round(box[4]))
        end_point = (round(box[5]), round(box[6]))
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        if OD:
            # update bicycle label if YOLO used
            if box[1] == 2:
                c = 'bicycle'
            else:
                c = classes[box[1]]
            cv2.putText(image, c, (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
        else:
            cv2.putText(image, classes[box[1]], (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color, thickness)  # + str(box[2]) + '%' for percentage
    return image


# function to generate event frame from array of coordinates
# slice_arr = Nx4 array with each row in format [epoch, x, y, polarity]
def generate_event_frame(slice_arr, folder=None, current_frame=None):
    # initialise image to desired resolution
    event_frame = np.full((480, 640, 3), 125, dtype=np.uint8)
    for row in slice_arr:
        # set pixel to black or white depending on polarity
        event_frame[row[2], row[1]] = 0 if row[3] == 0 else 255
    if current_frame is not None and folder is not None:
        # save frame
        if not os.path.isdir(folder):
            os.mkdir(folder)
        name = os.path.join(folder, 'frame' + str(current_frame) + '.jpg')
        print('Captured...' + name)
        cv2.imwrite(name, event_frame)
    return event_frame


# function to calculate bounding box coordinates from cluster coordinates
#  points = Nx2 array of pixel coordinates in cluster
def bounding_coordinates(points):
    x_coordinates, y_coordinates = zip(*points)
    min_x = min(x_coordinates)
    min_y = min(y_coordinates)
    max_x = max(x_coordinates)
    max_y = max(y_coordinates)
    return [min_x, min_y, max_x, max_y]


# function to crop frame to specified coordinates
# box_list = list of coordinates in format [[x1, y1, x2, y2],...]
# default pixel_error adds 100 pixels to each side of box
# returns list of cropped images
def crop_frame(box_list, frame, folder=None, current_frame=None, pixel_error=100):
    cropped_regions = []
    count = 0
    for box in box_list:
        count += 1
        box = [round(p) for p in box]
        # determine areas to crop to
        min_x = max(0, box[0] - pixel_error)
        max_x = min(box[2] + pixel_error, 4095)
        min_y = max(0, box[1] - pixel_error)
        max_y = min(2159, box[3] + pixel_error)

        cropped_img = frame[min_y:max_y, min_x:max_x]  # crop image
        cropped_regions.append(cropped_img)
        if current_frame is not None and folder is not None:
            # save image
            if not os.path.isdir(folder):
                os.mkdir(folder)
            name = os.path.join(folder, 'f' + str(current_frame) + 'crop' + str(count) + '.jpg')
            cv2.imwrite(name, cropped_img)
    return cropped_regions
