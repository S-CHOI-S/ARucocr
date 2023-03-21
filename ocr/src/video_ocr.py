#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import rospy
# import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String
# set ff = unix
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
#pytesseract.pytesseract.tesseract_cmd = '/home/shjung/.local/bin/pytesseract'
#img = cv2.imread("/Users/shjun/Desktop/digit.png")
#imchar = pytesseract.image_to_string(img)
# print(imchar)

# 언어도 설정하려면
# text = pytesseract.image_to_string(img, lang=None)
# print(text)

# imgH, imgW, _ = img.shape
# imgbox = pytesseract.image_to_boxes(img)
# for boxes in imgbox.splitlines():
#     boxes = boxes.split(' ')
#     x, y, w, h = int(boxes[1]), int(boxes[3]), int(boxes[3]), int(boxes[4])
#     cv2.rectangle(img, (x, imgH-y), (w, imgH-h), (0, 255, 0), 3)

pub = rospy.Publisher('ocr', String, queue_size=10)
rospy.init_node('video_ocr', anonymous=True)
rate = rospy.Rate(1)


# video
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot not read video file')

cntr = 0
while not rospy.is_shutdown():
    ret, frame = cap.read()
    cntr += 1
    # if ((cntr % 20) == 0):
    imgH, imgW, _ = frame.shape
    # x1, y1, w1, h1 = 0, 0, imgH, imgW
    # imchar = pytesseract.image_to_string(frame)
    # imgboxes = pytesseract.image_to_boxes(frame)
    # for boxes in imgboxes.splitlines():
    #     boxes = boxes.split(' ')
    #     x, y, w, h = int(boxes[1]), int(
    #         boxes[3]), int(boxes[3]), int(boxes[4])
        #cv2.putText(frame, imchar, (x1+int(w1/50), y1), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0, 255, 0), 3)
        #font = cv2.FONT_HERSHEY_SIMPLEX.as_integer_ratio
    cv2.imshow('Video detection', frame)
    # print(imchar)
    imchar = "여우와 호랑이"
    if imchar.find("여우")!= -1:
        print("여우")
        pub.publish("여우")
    elif imchar.find("늑대"):
        print("늑대")
        pub.publish("늑대")
    elif imchar.find("쥐"):
        print("쥐")
        pub.publish("쥐")
    else:
        print("None")
        pub.publish("None")
    # pub.publish(imchar)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

                
# cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()