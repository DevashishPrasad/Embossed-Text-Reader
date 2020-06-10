import cv2 as cv2
import numpy as np
import imutils
import pytesseract

# Capture from web cam
cap = cv2.VideoCapture(0)

_,image = cap.read() 
# You can load a saved file from disk
# image = cv2.imread('test13.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(img, 20, 50)# Adjust these parameters according to your image
dilate = cv2.dilate(edged, None, iterations=1)
# We don't perform erosion, it completely depends on the image and need
# erode = cv2.erode(dilate, None, iterations=1)

# make an empty mask 
mask = np.ones(img.shape[:2], dtype="uint8") * 255

# find contours
cnts,hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

orig = img.copy()
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 200:
        cv2.drawContours(mask, [c], -1, 0, -1)
    
    x,y,w,h = cv2.boundingRect(c)

    # Filter and remove more contours according to your need
    if(w>h):
        cv2.drawContours(mask, [c], -1, 0, -1)
        
# Remove those ignored contours
newimage = cv2.bitwise_and(dilate.copy(), dilate.copy(), mask=mask)
# Dilate again if necessary
img2 = cv2.dilate(newimage, None, iterations=1)
ret2,th1 = cv2.threshold(newimage,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# tesseract on the filtered image
temp = pytesseract.image_to_string(th1)
# Write output on the image
cv2.putText(image,temp,(50,100),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),3)

# Show the results
cv2.imshow('Original image', cv2.resize(image,(640,480)))
cv2.imshow('Dilated', cv2.resize(dilate,(640,480)))
cv2.imshow('New Image', cv2.resize(newimage,(640,480)))
cv2.imshow('Inverted Threshold', cv2.resize(th1,(640,480)))

cv2.waitKey(0)
cv2.destroyAllWindows()

