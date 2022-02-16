#!/usr/bin/python3
import os
import sys
import cv2
import time
import numpy as np

# Path of external libraries
#sys.path.insert(0, "/opt/ros/melodic/lib/python2.7/dist-packages/rospy/")
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from calibration_store import load_coefficients
from vision_message.msg import image_data as Pixel


# Global variables for for matrices of each camera left camera
path_l = '/home/fer/Control_servo_visual/Code/Visual_Servoing_Drone/Webots_controllers/left_parameters.yml'
mtx_l, dist_l = load_coefficients(path_l)

# Global variables for for matrices of each camera left camera
path_r = '/home/fer/Control_servo_visual/Code/Visual_Servoing_Drone/Webots_controllers/right_parameters.yml'
mtx_r, dist_r = load_coefficients(path_r)

## Gloabl variable fot the previes frame


class ImageConverter:
    def __init__(self):
        # Definition of bridge
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/Pioner_3dx/camera_lr/", Image, self.callback)

        # Definition of aruco markers parameters
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.parameters = parameters = cv2.aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshConstant = 10

        # Definition of matrices l
        self.mtx_l = mtx_l
        self.dist_l = dist_l

        # Definition of matrices r
        self.mtx_r = mtx_r
        self.dist_r = dist_r

        # baseline camera
        self.B = 0.2

        # Description os publisher
        self.pixel_pub = rospy.Publisher("Pioner_3dx/Pixels_data", Pixel, queue_size=10)
        self.pixel_message = Pixel()
        
        ## Variables for previous frame
        self.k = 0
        self.prev_frame = np.zeros((400,500), dtype=np.uint8)

        ## Create detector

        self.orb = cv2.ORB_create()
        self.sift = cv2.xfeatures2d.SIFT_create()

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Section to divide the image
        h = cv_image.shape[0]
        w = cv_image.shape[1]
        half_w = int(w / 2)

        # Get two Images from node
        img_l = self.correction(cv_image[0:h, 0:half_w], self.mtx_l, self.dist_l)
        img_r = self.correction(cv_image[0:h, half_w:w], self.mtx_r, self.dist_r)

        # Get values of aruco and draw corners
        gray_l, corner_l = self.aruco_detection(img_l)
        gray_r, corner_r = self.aruco_detection(img_r)

        ## Section to obtain the frames
        if self.k>0:
            #traslation = self.visual_odometry_orb(gray_l, self.prev_frame)
            traslation = self.visual_odometry_sift(gray_l, self.prev_frame)
            print(traslation)


            
        #corner_l = self.better_point(corner_l)
        #self.show(gray_l, gray_r)

        # Draw circles on corners
        #final_l = self.draw_points(corner_l, gray_l)
        #final_r = self.draw_points(corner_r, gray_r)
        
        # Send Values to Node
        #system_coordinates = self.point_3d(corner_l, corner_r,self.mtx_l, self.mtx_r, self.B)
        #self.send_message(system_coordinates)
        self.prev_frame = gray_l 
        # Frame
        self.k = 1

    def visual_odometry_orb(self, frame, prev_frame):
        fx = self.mtx_l[0,0]
        cx = self.mtx_l[0,2]
        cy = self.mtx_l[1,2]
        fx = fx*0.0774
        

        
        kp1, des1 = self.orb.detectAndCompute(prev_frame, None)
        kp2, des2 = self.orb.detectAndCompute(frame, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key=lambda x: x.distance)
        img_matching = cv2.drawMatches(prev_frame, kp1, frame, kp2, matches[0:50], None)
        cv2.imshow('feature matching', img_matching)
        cv2.waitKey(3)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        # Calcular la Matriz Esencial
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC,prob=0.999, threshold=1)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))
        return t

    def visual_odometry_sift(self, frame, prev_frame):
        fx = self.mtx_l[0,0]
        cx = self.mtx_l[0,2]
        cy = self.mtx_l[1,2]
        fx = fx*0.0774
        

        
        kp1, des1 = self.sift.detectAndCompute(prev_frame, None)
        kp2, des2 = self.sift.detectAndCompute(frame, None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
        matches = bf.match(des2,des1)
        matches = sorted(matches, key=lambda x: x.distance)
        img_matching = cv2.drawMatches(prev_frame, kp1, frame, kp2, matches[0:50], None)
        cv2.imshow('feature matching', img_matching)
        cv2.waitKey(3)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        # Calcular la Matriz Esencial
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC,prob=0.999, threshold=1)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))
        return t


    def better_point(self, corner):
        # Convert corner to better vector
        corner = corner.T
        
        # Fet just values of interes
        vect_1 = corner[0:2, 0]
        vect_4 = corner[0:2, 3]

        # Get distance
        aux = vect_4 - vect_1
        d = np.linalg.norm(aux, 2)
        T = d/3

        # Get angle value for 
        angle = np.arctan2(aux[0], aux[1])

        corner[0, 0] = corner[0, 0] + T*np.sin(angle)
        corner[1, 0] = corner[1, 0] + T*np.cos(angle)

        corner[0, 3] = corner[0, 0] + T*np.sin(angle)
        corner[1, 3] = corner[1, 0] + T*np.cos(angle)
        
        return corner.T


    def show(self, img_l, img_r):
        cv2.imshow("lef", img_l)
        cv2.imshow("right", img_r)
        cv2.waitKey(3)

    def draw_points(self, corners, img):
        # Convert corner to normal vector
        corners =  corners.T

        # Define Radius of the circle
        r = 5

        #  Define color of the circles
        color = (255, 0, 0)

        # Width circle
        thickness = 2

        for k in range(0, corners.shape[1]):
            center_coordinates = (int(corners[0, k]), int(corners[1, k]))
            final = cv2.circle(img, center_coordinates, r, color, thickness)
        return final


    def aruco_detection(self, frame_l):
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        corner_l, ids_l, reject_l = cv2.aruco.detectMarkers(gray_l, self.dictionary, parameters=self.parameters)
        if np.all(ids_l is not None):
            cv2.aruco.drawDetectedMarkers(gray_l, corner_l)
            corner_l = np.array(corner_l).reshape(4, 2)
        else:
            corner_l = np.array([])
        return gray_l, corner_l


    def correction(self, frame, mtx, dst):
        # Obtain dimension of the image
        h_frame = frame.shape[0]
        w_frame = frame.shape[1]
        new_camera, roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (w_frame, h_frame), 1, (w_frame, h_frame))
        # Get no distortion image
        frame_u = cv2.undistort(frame, mtx, dst, None, new_camera)
        return frame_u

    def center_camera(self, pixels, mtx):
        # Definition of the center points of the camera
        uo = mtx[0, 2]
        vo = mtx[1, 2]

        pixel = np.array([pixels[:4, 0], pixels[:4, 1], [1, 1, 1, 1]])
        transformation = np.array([[1, 0, -uo], [0, 1, -vo], [0, 0, 1]])
        center = transformation @ pixel
        center = center[0:2, 0:4]
        return center

    def send_message(self, message):
        if message.size != 0:

            self.pixel_message.u1 = int(message[0, 0])
            self.pixel_message.v1 = int(message[1, 0])
            self.pixel_message.x1 = message[2, 0]
            self.pixel_message.y1 = message[3, 0]
            self.pixel_message.z1 = message[4, 0]

            self.pixel_message.u2 = int(message[0, 1])
            self.pixel_message.v2 = int(message[1, 1])
            self.pixel_message.x2 = message[2, 1]
            self.pixel_message.y2 = message[3, 1]
            self.pixel_message.z2 = message[4, 1]

            self.pixel_message.u3 = int(message[0, 2])
            self.pixel_message.v3 = int(message[1, 2])
            self.pixel_message.x3 = message[2, 2]
            self.pixel_message.y3 = message[3, 2]
            self.pixel_message.z3 = message[4, 2]

            self.pixel_message.u4 = int(message[0, 3])
            self.pixel_message.v4 = int(message[1, 3])
            self.pixel_message.x4 = message[2, 3]
            self.pixel_message.y4 = message[3, 3]
            self.pixel_message.z4 = message[4, 3]

            self.pixel_message.flag = 1
        else:
            self.pixel_message.flag = 0

        self.pixel_pub.publish(self.pixel_message)

    def point_3d(self, pixel_l, pixel_r, mtxl, mtxr, B):
        if pixel_l.size != 0 and pixel_r.size != 0:
            # Parameters lef camera intrinsic parameters
            fx_l = mtxl[0, 0]
            fy_l = mtxl[1, 1]
            aux_pixel_l = self.better_point(pixel_l)
            aux_pixel_r = self.better_point(pixel_r)

            # Calc of disparity between point
            center_l = self.center_camera(aux_pixel_l, mtxl)
            center_r = self.center_camera(aux_pixel_r, mtxl)
            disparity = center_l - center_r

            z = np.array([[0.0, 0.0, 0.0, 0.0]])
            x = np.array([[0.0, 0.0, 0.0, 0.0]])
            y = np.array([[0.0, 0.0, 0.0, 0.0]])

            # Conver_pixel trasnpose
            pixel_l = aux_pixel_l.T

            for k in range(0, disparity.shape[1]):
                z[0, k] = 0.0774 * (B * fx_l) / disparity[0, k]
                x[0, k] = (z[0, k] * center_l[0, k]) / (fx_l*0.0774)
                y[0, k] = (z[0, k] * center_l[1, k]) / (fx_l*0.0774)

            position = np.array([pixel_l[0, 0:4], pixel_l[1, 0:4], x[0, 0:4], y[0,:4], z[0, 0:4]]).reshape(5, 4)
        else:
            position = np.array([])
        return position


def main(args):
    rospy.init_node('Opencv_process_Pionner_3dx', anonymous=False)
    ic = ImageConverter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
