# import libraries:
from itertools import count
import cv2
from matplotlib.pyplot import contour
import numpy as np
from pyrsistent import v
import cv2.aruco as aruco
import mask
import aruco_markers
import imageCutter

# import CVtask image and resize it:
img = cv2.imread("C:/Users/adity/OneDrive/Documents/ATULAYA/CVtask/task_images/CVtask.jpg")
img = cv2.resize(img, None, fx=0.5, fy=0.5)

# showing CVtsk img. 
cv2.imshow("CVtask", img)                                              

# HSV value range of following color from built mask library using trackbar:
green = np.array([[32,32,150], [60,255,255]])
orange = np.array([[8,180,14], [100,255,255]])
black= np.array([[0,0,0], [0,0,0]])
pinkpeach = np.array([[0,15,0], [60,29,239]])

color ={1:green, 2:orange, 3:black, 4:pinkpeach}       # DICT of ids and color, ids value aruco fill in that color.
mask_contours = mask.mask(img, color)                  # mask_contours are DICT of contour with their ids w.r.t color.

# import aruco markers:
aruco_marker1 = cv2.imread("C:/Users/adity/OneDrive/Documents/ATULAYA/CVtask/task_images/Ha.jpg")
aruco_marker2 = cv2.imread("C:/Users/adity/OneDrive/Documents/ATULAYA/CVtask/task_images/HaHa.jpg")
aruco_marker3 = cv2.imread("C:/Users/adity/OneDrive/Documents/ATULAYA/CVtask/task_images/LMAO.jpg")
aruco_marker4 = cv2.imread("C:/Users/adity/OneDrive/Documents/ATULAYA/CVtask/task_images/XD.jpg")

# find out corners, ids of aruco markers:
(corners1, ids1, rejected1) = aruco_markers.Findaruco(aruco_marker1)
(corners2, ids2, rejected2) = aruco_markers.Findaruco(aruco_marker2)
(corners3, ids3, rejected3) = aruco_markers.Findaruco(aruco_marker3)
(corners4, ids4, rejected4) = aruco_markers.Findaruco(aruco_marker4)

# convert into integer values and return also angle by which img tilted:
corners1, ids1, angle1 = aruco_markers.int_corners_ids(corners1, ids1)
corners2, ids2, angle2 = aruco_markers.int_corners_ids(corners2, ids2)
corners3, ids3, angle3 = aruco_markers.int_corners_ids(corners3, ids3)
corners4, ids4, angle4 = aruco_markers.int_corners_ids(corners4, ids4)
corners = np.array([corners1, corners2, corners3, corners4])          # making a np array of all corners.

# Making a arucoDICT of markers with their ids value:
aruco_dict = {ids1:[angle1, aruco_marker1], ids2:[angle2, aruco_marker2], ids3:[angle3,aruco_marker3], ids4:[angle4, aruco_marker4]}

approx = {}                                                           # creating a empty DICT:

# for loop for containing only square countour into approxDICT with their ids:
for (ids) in (mask_contours):
    x,y,w,h = cv2.boundingRect(mask_contours[ids])

    #aspect_ratio is the ratio of adjacent sides:
    aspect_ratio = float(w)/h
    if ((0.95 <= aspect_ratio) & (aspect_ratio <= 1.05)):
        approx[ids] = mask_contours[ids]   


# for loop for updating aruco img(remove outsides white space) and pasting into CVtask img:
for ids in aruco_dict:
    pt1 = approx[ids]
    angle = aruco_dict[ids][0]
    aruco_img = aruco_dict[ids][1]

    if(ids == 1):
        aruco_img, pt2 = imageCutter.remove_extra_padding(aruco_img, angle)

    elif(ids == 2):
        aruco_img, pt2 = imageCutter.remove_extra_padding(aruco_img, angle)

    elif(ids == 3):
        aruco_img, pt2 = imageCutter.remove_extra_padding(aruco_img, angle)  
    
    elif(ids == 4):
        aruco_img, pt2 = imageCutter.remove_extra_padding(aruco_img, angle)

    matrix, _ = cv2.findHomography(pt2, pt1) 
    warp_img =cv2.warpPerspective(aruco_img, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pt1, (0,0,0))
    img = img + warp_img

# showing final img:    
cv2.imshow("taskcomplete", img)
cv2.waitKey(0)
cv2.destroyAllWindows()    


