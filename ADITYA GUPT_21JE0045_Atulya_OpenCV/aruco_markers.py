import cv2
import numpy as np
import cv2.aruco as aruco
import math

#function for calculating corners & ids of aruco_markers:
def Findaruco(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                   
    key = getattr(aruco, f'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create() 

    (corners, ids, rejected) = aruco.detectMarkers(img, arucoDict, parameters = arucoParam)
    cv2.imwrite(f"{int(ids)}.jpg",img)                   # saving the aruco img with their id name.
    return (corners, ids, rejected) 

# fuction for converting corners & ids type into integer: 
def int_corners_ids(corners, ids):
    for(markercorner, markerid) in zip(corners, ids):
        corners = markercorner.reshape((4,2))
        (topleft, topright, bottomright, bottomleft) = corners

        topleft = (int(topleft[0]), int(topleft[1]))
        topright = (int(topright[0]), int(topright[1]))
        bottomright = (int(bottomright[0]), int(bottomright[1]))
        bottomleft = (int(bottomleft[0]), int(bottomleft[1]))
    
# (cx, cy) is the center of the aruco:
        cx = int((bottomleft[0] + topright[0])/2.0)
        cy = int((bottomleft[1] + topright[1])/2.0)

# (mx, my) is the mid point of left side to the centre:
        mx = int((bottomleft[0] + topleft[0])/2.0)
        my = int((bottomleft[1] + topleft[1])/2.0)

        val = ((my-cy)/(mx-cx))

# angle by which img is rotated into clockwise:
        angle = math.degrees(math.atan(val))
    
    ids = int(ids)
    
    return (corners, ids, angle)







