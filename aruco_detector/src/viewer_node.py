#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('unibas_face_detector')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class viewer:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/unibas_face_detector/faces", Image, self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)    

    cv2.imshow("faces", cv_image)
    cv2.waitKey(30)
   

def main(args):
  v = viewer()
  rospy.init_node('viewer_node', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


""" 
import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS,30)

rospy.init_node("webcam_pub",anonymous=True)
image_pub = rospy.Publisher("webcam_image",Image,queue_size=1)

bridge = CvBridge()

while not rospy.is_shutdown():
  ret, cv_image = cap.read()

  image_pub.publish(bridge.cv2_to_imgmsg(cv_image,"bgr8"))

cap.release()
cv2.destroyAllWindows() """