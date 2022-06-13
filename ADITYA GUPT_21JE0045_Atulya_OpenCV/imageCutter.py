import numpy as np
import cv2
import cv2.aruco as aruco

# function for rotation and slicing of aruco_markers:
def remove_extra_padding(img, angle):
    h,w = img.shape[:-1]
    
    # rotation point (centre) of the img;
    rot_point = w//2, h//2

    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)          # ing converted into HSV from color.
    rot_mat = cv2.getRotationMatrix2D(rot_point,angle,1)   # gives a matrix in which rotation in anticlockwise with given angle about rot_point.    
    rot = cv2.warpAffine(img_hsv,rot_mat,(w,h))            # rotate the hsv_img using rot_matrix.
    img2 = cv2.cvtColor(rot, cv2.COLOR_HSV2BGR)            # now, img converted into bgr from hsv.

# find out corners, ids of the img:
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create() 

    (corners, ids, rejected) = aruco.detectMarkers(img2, arucoDict, parameters = arucoParam)

# converting corners & ids type into integer: 
    for corner in corners:    
        corner = corner.reshape((4,2))
        (topleft, topright, bottomright, bottomleft) = corner

        topleft = (int(topleft[0]), int(topleft[1]))
        topright = (int(topright[0]), int(topright[1]))
        bottomright = (int(bottomright[0]), int(bottomright[1]))
        bottomleft = (int(bottomleft[0]), int(bottomleft[1]))

# slicing of img(remove the extra padding):
    crop_img = img2[topleft[0]:topright[0], topleft[1]:bottomleft[1]]

# create a numpy of corners of the img:    
    crop_cr = np.array([[0,0], [crop_img.shape[1],0], [crop_img.shape[1],crop_img.shape[0]], [0,crop_img.shape[0]]])
    
    return (crop_img, crop_cr)   
