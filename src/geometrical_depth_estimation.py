#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import math
from sensor_msgs.msg import Image
from yolov3_trt_ros.msg import BoundingBox, BoundingBoxes

calibration_image = np.empty(shape=[0])
bbox_list_raw = []
CAMERA_HEIGHT = 0.16
FOV_H = 55 #need to change
FY = 346.5049
CY = 204.58251


def image_callback(data):
    global calibration_image
    calibration_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    #calibration_image = cv2.cvtColor(calibration_image, cv2.COLOR_RGB2BGR)


def bbox_callback(data):
    global bbox_list_raw
    bbox_list_raw = []
    for bbox in data.bounding_boxes:
        bbox.xmin = max(min(639, bbox.xmin), 0)
        bbox.ymin = max(min(479, bbox.ymin), 0)
        bbox.xmax = max(min(639, bbox.xmax), 0)
        bbox.ymax = max(min(479, bbox.ymax), 0)
        box = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, 0, 0, 0]
        bbox_list_raw.append(box)


def get_depth(bbox_list, camera_height, fy, cy):
    for index, bbox in enumerate(bbox_list):
        # normalized Image plane
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        y_norm = (ymax - cy) / fy
        y_distance = 1 * camera_height

        delta_x = (xmax+xmin)/2 - 320
        azimuth = (delta_x/320) * FOV_H
        y_distance = 1 * camera_height / y_norm
        x_distance = 100 * (y_distance * math.tan(math.pi * (azimuth/180.)))
        y_distance = 100 * (y_distance - 0.15) * 0.5826 + 2.6533
        distance = int(math.sqrt((x_distance * x_distance) + (y_distance * y_distance)))

        #distance = 1 * camera_height / y_norm
        # m -> cm
        bbox_list[index][4] = distance
        bbox_list[index][5] = x_distance
        bbox_list[index][6] = y_distance
    return bbox_list


def draw_box_and_depth(image, bbox_list):
    for bbox in bbox_list:
        print("x_dist = ",{bbox[5]})
        print("y_dist = ", {bbox[6]})

        x1, y1, x2, y2, depth, x_distance, y_distance = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, f"{depth}cm", (x2, y2+10), 1, 1, (255, 255, 0), 2)
    cv2.imshow('result', image)
    cv2.waitKey(1)


def start_depth_estimation():
    rate = rospy.Rate(10)
    image_sub = rospy.Subscriber('/usb_cam/cailbration_image', Image, image_callback)
    bbox_sub = rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, bbox_callback)
    while not rospy.is_shutdown():
        rate.sleep()
        if calibration_image.shape[0] == 0:
            continue
        bbox_list = get_depth(bbox_list_raw, CAMERA_HEIGHT, FY, CY)
        draw_box_and_depth(calibration_image, bbox_list)


if __name__ == '__main__':
    rospy.init_node('geometrical_depth_estimation', anonymous=True)
    start_depth_estimation()
