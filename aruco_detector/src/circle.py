#!/usr/bin/env python3

import numpy as np
import cv2
import rospy

method ='MOG2' # or 'KNN'

min_H,max_H=160,206
min_S,max_S=149,185
min_V,max_V=53,227
def onChange(x):
    global minBounds, maxBounds
    minBounds[0]= cv2.getTrackbarPos('minH', 'Settings')
    minBounds[1] = cv2.getTrackbarPos('minS', 'Settings')
    minBounds[2]= cv2.getTrackbarPos('minV', 'Settings')
    maxBounds[0] = cv2.getTrackbarPos('maxH', 'Settings')
    maxBounds[1] = cv2.getTrackbarPos('maxS', 'Settings')
    maxBounds[2] = cv2.getTrackbarPos('maxV', 'Settings')
cv2.namedWindow("Settings")
cv2.createTrackbar('maxH', 'Settings', max_H, 255, onChange)
cv2.createTrackbar('minH', 'Settings', min_H, 255, onChange)
cv2.createTrackbar('maxS', 'Settings', max_S, 255, onChange)
cv2.createTrackbar('minS', 'Settings', min_S, 255,onChange)
cv2.createTrackbar('maxV', 'Settings', max_V, 255, onChange)
cv2.createTrackbar('minV', 'Settings', min_V, 255, onChange)
minBounds = np.array([min_H, min_S, min_V])
maxBounds = np.array([max_H, max_S, max_V])

if method=='KNN':
    bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    blk1,blk2=(3, 3),(3, 3)
elif method=='MOG2':
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    blk1,blk2=(3, 3),(9, 9)

history_length = 20
bg_subtractor.setHistory(history_length)
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, blk1)
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, blk2)

img = cv2.imread('/home/choisol/catkin_ws/src/unibas_face_detector/images/manipulator.jpg')
img = cv2.resize(img,dsize=(0,0),fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA)
while True:
    img_blur = cv2.medianBlur(img,7)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, minBounds, maxBounds)
    res = cv2.bitwise_and(img, img, mask=mask)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,150,param1=50,param2=33,minRadius=10,maxRadius=100)
    edges = cv2.Canny(gray,80,200,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    lines =lines.reshape((-1,2))
    for rho, theta in lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
    # if circles is not None:
    #     circles = np.round(circles[0,:]).astype('int')
    #     for (x,y,r) in circles:
    #         cv2.circle(img, (x,y), r, (0,0,255), 4)
    #         cv2.rectangle(img, (x-4, y-4), (x+4,y+4), (0,128,255), -1)
    #         print(x,y,r)

    # cv2.imshow('detected circles',img)
    cv2.imshow('frame', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    cv2.imshow('canny',edges)
    key = cv2.waitKey(1)
    if key == 27:
        break
 
key = cv2.waitKey(0) & 0xFF
if key == ord('q'):
    cv2.destroyAllWindows()