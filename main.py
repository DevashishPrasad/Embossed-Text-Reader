import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract

image = cv2.imread('test13.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(img, 40, 90)
dilate = cv2.dilate(edged, None, iterations=2)
# erode = cv2.erode(dilate, None, iterations=1)

# ret2,th1 = cv2.threshold(dilate,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# ?th1 = cv2.GaussianBlur(dilate, (7,7), 0)

mask = np.ones(img.shape[:2], dtype="uint8") * 255

cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

orig = img.copy()
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 300:
        cv2.drawContours(mask, [c], -1, 0, -1)
    
    x,y,w,h = cv2.boundingRect(c)

    if(w>h):
        cv2.drawContours(mask, [c], -1, 0, -1)
    # draw the contours on the image

    # cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    # cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)

newimage = cv2.bitwise_and(dilate.copy(), dilate.copy(), mask=mask)
img2 = cv2.dilate(newimage, None, iterations=3)
ret2,th1 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

temp = pytesseract.image_to_string(th1)
cv2.putText(image,temp,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.8,(0,255,255),3)

cv2.imshow('Original image', cv2.resize(image,(640,480)))
cv2.imshow('Dilated', cv2.resize(dilate,(640,480)))
cv2.imshow('New Image', cv2.resize(newimage,(640,480)))
cv2.imshow('Inverted Threshold', cv2.resize(th1,(640,480)))
# cv2.imshow()

cv2.waitKey(0)

cv2.destroyAllWindows()

