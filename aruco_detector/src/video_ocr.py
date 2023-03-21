#!/usr/bin/env python
import roslib
roslib.load_manifest('unibas_face_detector')

import sys
import rospy
import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = "/usr/share/tesseract-ocr"

#img = cv2.imread("/Users/shjun/Desktop/digit.png")
#imchar = pytesseract.image_to_string(img)
#print(imchar)

# 언어도 설정하려면
# text = pytesseract.image_to_string(img, lang=None)
# print(text)

# imgH, imgW, _ = img.shape
# imgbox = pytesseract.image_to_boxes(img)
# for boxes in imgbox.splitlines():
#     boxes = boxes.split(' ')
#     x, y, w, h = int(boxes[1]), int(boxes[3]), int(boxes[3]), int(boxes[4])
    
#     cv2.rectangle(img, (x, imgH-y), (w, imgH-h), (0, 255, 0), 3)
    
#video
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     cap =cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError('Cannot not read video file')

cntr = 0

while not rospy.is_shutdown():
    ret, frame = cap.read()
    cntr += 1
    if((cntr%20) == 0):
        imgH, imgW, _ = frame.shape
        x1, y1, w1, h1 = 0, 0, imgH, imgW
        imchar = pytesseract.image_to_string(frame)
        imgboxes = pytesseract.image_to_boxes(frame)
        
        for boxes in imgboxes.splitlines():
            boxes = boxes.split(' ')
            x, y, w, h = int(boxes[1]), int(boxes[3]), int(boxes[3]), int(boxes[4])
    
            #cv2.putText(frame, imchar, (x1+int(w1/50), y1), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX.as_integer_ratio
            cv2.imshow('Video detection', frame)
            print(imchar)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

# def main():
#     print("Initializing ROS-node")
#     rospy.init_node('video_ocr', anonymous=True)
#     print("Ready to read words!")
#     ImageConverter()
#     rospy.sleep(1)
#     rospy.spin()

# if __name__ == '__main__':
#     main()
