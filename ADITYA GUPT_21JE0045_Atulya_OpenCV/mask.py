import cv2
import numpy as np

# comment out code for calculating HSV range for color:

# img = cv2.imread("C:/Users/adity/OneDrive/Documents/ATULAYA/computer vision/CVtask.jpg")
# img = cv2.resize(img,None,fx=0.5,fy=0.5)
# cv2.imshow("original",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.namedWindow("trackbar")
# cv2.resizeWindow("trackbar",640,240)
# def any(x):
#     pass
# cv2.createTrackbar("hue min","trackbar",0,179,any)
# cv2.createTrackbar("hue max","trackbar",179,179,any)
# cv2.createTrackbar("sat min","trackbar",0,255,any)
# cv2.createTrackbar("sat max","trackbar",255,255,any)
# cv2.createTrackbar("val min","trackbar",0,255,any)
# cv2.createTrackbar("val max","trackbar",255,255,any)

# while True:
#     h_min=cv2.getTrackbarPos("hue min","trackbar")
#     h_max=cv2.getTrackbarPos("hue max","trackbar")
#     sat_min=cv2.getTrackbarPos("sat min","trackbar")
#     sat_max=cv2.getTrackbarPos("sat max","trackbar")
#     val_min=cv2.getTrackbarPos("val min","trackbar")
#     val_max=cv2.getTrackbarPos("val max","trackbar")
#     img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     lower = np.array([h_min,sat_min,val_min])
#     upper = np.array([h_max,sat_max,val_max])
#     mask = cv2.inRange(img_hsv,lower,upper)
#     bitand = cv2.bitwise_and(img,img,mask=mask)
#     cv2.imshow("original",img)
#     cv2.imshow("hsv",img_hsv)
#     cv2.imshow("mask",mask)
#     cv2.imshow("segmented",bitand)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cv2.destroyAllWindows()

# HSV value of following color:
    # green = np.array([[32,32,150], [60,255,255]])
    # orange = np.array([[8,180,14], [100,255,255]])
    # black= np.array([[0,0,0], [0,0,0]])
    # pinkpeach = np.array([[0,15,0], [60,29,239]])

# function for calculating countoursDICT with ids of image:
def mask(img, color):
    colorarr = color.values()
    colorids = color.keys()
    color_mask = {}
    mask_contours = {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                       # img converted into hsv from bgr.

# for loop for finding mask of the different color:
    for (color, ids) in zip(colorarr, colorids):
        mask = cv2.inRange(hsv, color[0], color[1])

    # using medianBlur remove noise of the img:    
        mask = cv2.medianBlur(mask, 5)  
        color_mask[ids] = mask  

# for loop for finding coutour of the img:
    for (img, ids) in zip(color_mask.values(), color_mask.keys()):
        cont, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # for loop for using approxPolyDP by which contours makes perfect:     
        for cnt in cont:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            mask_contours[ids] = approx
      
    return (mask_contours)
    

