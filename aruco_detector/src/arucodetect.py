#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('unibas_face_detector')

import sys
import cv2
import numpy as np
import imutils
import math
import argparse
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco

# hardcoded intrinsic matrix for my webcam
A = [[1019.37187, 0, 618.709848], [0, 1024.2138, 327.280578], [0, 0, 1]] 
A = np.array(A)

# constants specific to the implementation with tracking
lk_params = dict( winSize  = (19, 19),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.01,
                       minDistance = 8,
                       blockSize = 19 )

FREQUENCY = 100 # of finding the aruco marker from scratch
TRACKING_QUALITY_THRESHOLD_PERCENTAGE = 100 #reducing this number will make the program tolerate poorer tracking without refreshing to fix it

imchar = None

def get_extended_RT(A, H):
	#finds r3 and appends
	# A is the intrinsic mat, and H is the homography estimated
	H = np.float64(H) #for better precision
	A = np.float64(A)
	R_12_T = np.linalg.inv(A).dot(H)

	r1 = np.float64(R_12_T[:, 0]) #col1
	r2 = np.float64(R_12_T[:, 1]) #col2
	T = R_12_T[:, 2] #translation
	
	#ideally |r1| and |r2| should be same
	#since there is always some error we take square_root(|r1||r2|) as the normalization factor
	norm = np.float64(math.sqrt(np.float64(np.linalg.norm(r1)) * np.float64(np.linalg.norm(r2))))
	
	r3 = np.cross(r1,r2)/(norm)
	R_T = np.zeros((3, 4))
	R_T[:, 0] = r1
	R_T[:, 1] = r2 
	R_T[:, 2] = r3 
	R_T[:, 3] = T
	return R_T

def augment(img, obj, projection, template, scale = 4):
    # takes the captureed image, object to augment, and transformation matrix  
    #adjust scale to make the object smaller or bigger, 4 works for the fox

    h, w = template.shape
    vertices = obj.vertices
    img = np.ascontiguousarray(img, dtype=np.uint8)

    #blacking out the aruco marker
    a = np.array([[0,0,0], [w, 0, 0],  [w,h,0],  [0, h, 0]], np.float64 )
    imgpts = np.int32(cv2.perspectiveTransform(a.reshape(-1, 1, 3), projection))
    cv2.fillConvexPoly(img, imgpts, (0,0,0))

    #projecting the faces to pixel coords and then drawing
    for face in obj.faces:
        #a face is a list [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices]) #-1 because of the shifted numbering
        points = scale*points
        points = np.array([[p[2] + w/2, p[0] + h/2, p[1]] for p in points]) #shifted to centre 
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)#transforming to pixel coords
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])
        
    return img

class three_d_object:
    def __init__(self, filename_obj, filename_texture, color_fixed = False):
        self.texture = cv2.imread(filename_texture)
        self.vertices = []
        self.faces = []
        #each face is a list of [lis_vertices, lis_texcoords, color]
        self.texcoords = []

        for line in open(filename_obj, "r"):
            if line.startswith('#'): 
                #it's a comment, ignore 
                continue

            values = line.split()
            if not values:
                continue
            
            if values[0] == 'v':
                #vertex description (x, y, z)
                v = [float(a) for a in values[1:4] ]
                self.vertices.append(v)

            elif values[0] == 'vt':
                #texture coordinate (u, v)
                self.texcoords.append([float(a) for a in values[1:3] ])

            elif values[0] == 'f':
                #face description 
                face_vertices = []
                face_texcoords = []
                for v in values[1:]:
                    w = v.split('/')
                    face_vertices.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        face_texcoords.append(int(w[1]))
                    else:
                        color_fixed = True
                        face_texcoords.append(0)
                self.faces.append([face_vertices, face_texcoords])

def display(img, f = 1):
    #takes an image as input and scaling factor and displays
    img = scale(img, f)
    cv2.imshow('dummy', img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scale(image, f):
    #scales an image acc to f
    h = int(image.shape[0]*f)
    w = int(image.shape[1]*f)
    return cv2.resize(image, (w,h), interpolation = cv2.INTER_CUBIC )

def get_bit_sig(image, contour_pts, thresh = 127):
    ans = []

    #getting all the 4 corners of the quad
    a, b = contour_pts[0][0]
    c, d = contour_pts[1][0]
    e, f = contour_pts[3][0]
    g, h = contour_pts[2][0]

    for i in range(8):
        for j in range(8):
            #using bilinear interpolation to find the coordinate using fractional contributions of the corner 4 points
            f1 = float(i)/8 + 1./16 #fraction1
            f2 = float(j)/8 + 1./16 #fraction2

            #finding the intermediate coordinates 
            upper_x = (1-f1)*a + f1*(c)
            lower_x = (1-f1)*e + f1*(g)
            upper_y = (1-f1)*b + f1*d
            lower_y = (1-f1)*(f) + f1*(h)

            x = int( (1-f2)*upper_x + (f2)*lower_x )
            y = int( (1-f2)*upper_y + (f2)*lower_y )

            #thresholding
            if image[y][x] >= 127:
                ans.append(1)
            else:
                ans.append(0)
    return ans

def match_sig(sig1, sig2, thresh = 62):
    # print(sum([ (1- abs(a - b)) for a, b in zip(sig1, sig2)]))
    if sum([ (1- abs(a - b)) for a, b in zip(sig1, sig2)]) >= 62:
        return True
    else:
        return False

def find_pattern_aruco(image, aruco_marker, sigs):
    #converting image to black and white
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    
    #adaptive thresholding for robustness against varying lighting
    thresholded = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,10)
    h, w = aruco_marker.shape

    _, contours ,_= cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for cnt in contours : 
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) 
        if approx.shape[0]==4:
            x1 = approx[0][0][0] 
            x2 = approx[1][0][0]
            y1 = approx[0][0][1]
            y2 = approx[1][0][1]

            norm = (x1 - x2)**2 + (y1 - y2)**2
            #constraint on minimum edge size of quad
            if norm > 100:
                temp_sig = get_bit_sig(gray, approx)
                match1 = match_sig(sigs[0], temp_sig)
                match2 = match_sig(sigs[1], temp_sig)
                match3 = match_sig(sigs[2], temp_sig)
                match4 = match_sig(sigs[3], temp_sig)

                if (match1 or match2 or match3 or match4):
                    dst_pts = approx
                    if match1:
                        src_pts = np.array([[0,0],[0,w], [h,w], [h,0]])
                    if match2:
                        src_pts = np.array([[0,w], [h,w], [h,0], [0,0]])
                    if match3:
                        src_pts = np.array([[h,w],[h,0], [0,0], [0,w]])
                    if match4:
                        src_pts = np.array([[h,0],[0,0], [0,w], [h,w]])

                    # removed for consistency across both programs - with and without tracking
                    # cv2.drawContours(image, [approx], 0, (0, 0, 255), 2) #mark red outline for found marker 

                    return src_pts, dst_pts, True

    #reaching here implies nothing was found
    return None, None, False              

   
def find_homography_aruco(image, aruco_marker, sigs):
    src_pts, dst_pts, found = find_pattern_aruco(image, aruco_marker, sigs)
    H = None
    if found:
        H, mask = cv2.findHomography(src_pts.reshape(-1,1,2), dst_pts.reshape(-1,1,2), cv2.RANSAC,5.0)

    if H is None:
        return False, None
    else:
        return True, H

def callback(data):
    global imchar
    imchar = data.data

    


if __name__ == '__main__':
    print("Initializing ROS-node")
    rospy.init_node('detect_markers', anonymous=True)
    # ROS Subscriber
    rospy.Subscriber("/ocr", String, callback)

    marker_colored = cv2.imread('/home/choisol/catkin_ws/src/unibas_face_detector/images/m1.png')
    assert marker_colored is not None, "Could not find the aruco marker image file"
	#accounts for lateral inversion caused by the webcam
    marker_colored = cv2.flip(marker_colored, 1)

    marker_colored = cv2.resize(marker_colored, (480,480), interpolation = cv2.INTER_CUBIC)
    marker = cv2.cvtColor(marker_colored, cv2.COLOR_BGR2GRAY)

    print("trying to access the webcam")
    cv2.namedWindow("webcam")
    vc = cv2.VideoCapture(0)
    assert vc.isOpened(), "couldn't access the webcam"
	
    # obj = three_d_object('/home/choisol/catkin_ws/src/unibas_face_detector/images/3d_objects/low-poly-fox-by-pixelmannen.obj', None)
    obj = three_d_object('/home/choisol/catkin_ws/src/unibas_face_detector/images/3d_objects/rat.obj', None)
    h,w = h,w = marker.shape
	#considering all 4 rotations
    marker_sig1 = get_bit_sig(marker, np.array([[0,0],[0,w], [h,w], [h,0]]).reshape(4,1,2))
    marker_sig2 = get_bit_sig(marker, np.array([[0,w], [h,w], [h,0], [0,0]]).reshape(4,1,2))
    marker_sig3 = get_bit_sig(marker, np.array([[h,w],[h,0], [0,0], [0,w]]).reshape(4,1,2))
    marker_sig4 = get_bit_sig(marker, np.array([[h,0],[0,0], [0,w], [h,w]]).reshape(4,1,2))

    sigs = [marker_sig1, marker_sig2, marker_sig3, marker_sig4]

    rval, frame = vc.read()
    assert rval,"couldn't access the webcam"
    h2, w2,  _ = frame.shape

    h_canvas = max(h, h2)
    w_canvas = w + w2
	
    while rval:
        rval, frame = vc.read() #fetch frame from webcam
        key = cv2.waitKey(20)
        if key == 27: # Escape key to exit the program
            break

        rospy.Subscriber("/ocr", String, callback)
        print(imchar)

        # if imchar.find("fox") != -1:
        #     obj = three_d_object('/home/choisol/catkin_ws/src/unibas_face_detector/images/3d_objects/low-poly-fox-by-pixelmannen.obj', None)
        # elif imchar.find("rat") != -1:
        #     obj = three_d_object('/home/choisol/catkin_ws/src/unibas_face_detector/images/3d_objects/rat.obj', None)
        # else:
        #     obj = three_d_object('/home/choisol/catkin_ws/src/unibas_face_detector/images/3d_objects/pirate-ship-fat.obj', None)
        
        canvas = np.zeros((h_canvas, w_canvas, 3), np.uint8) #final display
        canvas[:h, :w, :] = marker_colored #marker for reference
        
        success, H = find_homography_aruco(frame, marker, sigs)
		# success = False
        if not success:
			# print('homograpy est failed')
            canvas[:h2 , w: , :] = np.flip(frame, axis = 1)
            cv2.imshow("webcam",canvas)
            continue

        R_T = get_extended_RT(A, H)
        transformation = A.dot(R_T)

        augmented = np.flip(augment(frame, obj, transformation, marker), axis = 1) #flipped for better control
        canvas[:h2 , w: , :] = augmented
        cv2.imshow("webcam", canvas)
        

    cv2.waitKey(0)
    cv2.destroyllWindows()
        
# rospy.sleep(1)
# rospy.spin()