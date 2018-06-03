import numpy as np
import cv2
from scipy.spatial import distance
#lower = np.array([0, 133, 100], dtype = "uint8")
#upper = np.array([255, 173, 127], dtype = "uint8")

#lower = np.array([0, 48, 80], dtype = "uint8")
#upper = np.array([20, 255, 255], dtype = "uint8")

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=20,
                                          varThreshold=16,
                                          detectShadows=False)
#fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)

while True:
    ret, frame = cap.read()


    fgmask = fgbg.apply(frame)
    #contours
    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(fgmask, contours, -1, (0, 255, 0), 3)
    if len(contours) > 0:
        # find largest contour in mask, use to compute minEnCircle
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        x_ = int(x - radius)
        y_ = int(y - radius)
        h_ = int(radius * 2)
        w_ = int(radius * 2)

        sss = distance.euclidean((x_,y_),(x_ +w_,y_ +h_))

        if sss>30:
            cv2.rectangle(fgmask, (x_, y_), (x_ + w_, y_ + h_), (254, 255, 0), 2)

    cv2.imshow('frame', fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
