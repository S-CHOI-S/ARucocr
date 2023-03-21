#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('unibas_face_detector')
import sys
import rospy
import cv2
import numpy as np

import math

from cv2 import aruco
from sensor_msgs.msg import Image
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

# def ptDist(p1,p2):
#   dx=p2[0]-p1[0]
#   dy=p2[1]-p1[1]
#   return math.sqrt(dx*dx+dy*dy)

# def ptMean(p1,p2):
#   return ((int(p1[0]+p2[0])/2, int(p1[1]+p2[1])/2))

# def rect2centerline(rect):
#   p0=rect[0]
#   p1=rect[1]
#   p2=rect[2]
#   p3=rect[3]
#   width=ptDist(p0,p1)
#   height=ptDist(p1,p2)

#   # centerline lies along longest median
#   if(height > width):
#     c1 = (ptMean(p0,p1), ptMean(p2,p3))
#   else:
#     c1 = (ptMean(p1,p2), ptMean(p3,p0))

#   return c1

# def showImage(img):
#   cv2.imshow('image', img)
#   cv2.waitKey(1)

# def callback(data):
#   rospy.loginfo(rospy.get_caller_id() + "I heard %s",data.data)

class videoWriter:
  def __init__(self,cap,fname,**kw):
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.fps = kw.get('fps',30)
    self.fcc = kw.get('fcc','DIVX')
    fourcc = cv2.VideoWriter_fourcc(*self.fcc)
    self.out = cv2.VideoWriter(fname,fourcc,self.fps,(int(width),int(height)))

  def update(self,frame):
    self.out.write(frame)

  def release(self):
    self.out.release()

def force_callback(data):
  forceX =  data.linear.x
  print(forceX)

def draw_axis(img, rotation_vec, t, K, dist, scale = 0.1):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array
    :rotation_vec - euler rotations, numpy array of length 3,
                    use cv2.Rodrigues(R)[0] to convert from rotation matrix
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :scale - factor to control the axis lengths
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    img = img.astype(np.float32)
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (0, 0, 255), 3)
    return img

# def img_callback(data):
#   global cv_img
#   cv_img = bridge.imgmsg_to_cv2(data,"CV_8UC3")

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
#cap.set(cv2.CAP_PROP_FPS,30)
vid = videoWriter(cap,'~/catkin_ws/src',fps=30,fcc='DIVX')


rospy.init_node("webcam_pub",anonymous=True)
rate = rospy.Rate(1000)

image_pub = rospy.Publisher("webcam_image",Image,queue_size=1)
pub = rospy.Publisher("chatter",String,queue_size=10)

rospy.loginfo('webcam_pub node started!')

bridge = CvBridge()

calib_data_path = "/home/choisol/AR/calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

marker_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

param_markers = cv2.aruco.DetectorParameters_create()

while not rospy.is_shutdown():
  ret, cv_image = cap.read()

  gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
  #gray_c = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
  #blur_gray = cv2.GaussianBlur(gray,(5,5),0)
  #edge_img = cv2.Canny(blur_gray,50,150)
  #cv_image = gray_c

  # marker_corners, marker_IDs, reject = cv2.aruco.detectMarkers(gray, marker_dict, parameters=param_markers)

  # if marker_corners:
  #       for ids, corners in zip(marker_IDs, marker_corners):

  #           corners = corners.reshape(4, 2)
  #           corners = corners.astype(int)
  #           draw_axis(cv_image, r_vectors, t_vectors, cam_mat, dist_coef)
  #           # print(ids, "  ", corners)

  # # draw circle
  # h,w,c = cv_image.shape
  # center=(w//2,h//2)
  # cv2.circle(cv_image,center,150,(0,0,255),5)
  
  vid.update(cv_image)

  image_pub.publish(bridge.cv2_to_imgmsg(cv_image,encoding="bgr8"))
  #image_pub.publish(bridge.cv2_to_imgmsg(edge_img,encoding="passthrough"))
  pub.publish("publish hello!")

  #rospy.Subscriber("chatter",String,callback)
  rospy.Subscriber("/estimated_force",Twist,force_callback)

  #rospy.spin()
  #rate.sleep()
  

cap.release()
vid.release()
cv2.destroyAllWindows()
