# Embossed-Text-Reader
This is a tesseract based OCR to read Embossed text on metals. It can also be use as a general OCR.

This is not generalized solution. The satements and parameters of some funtions will changed according to the texture, color, lighting effect and visibility of text of image.

Tesseract is an open-sourced OCR which is capable of reading text from papers, pdfs and other clean formats. Tesseract fails when tried to perform OCR on noisy and dirty images (for eg. Embossed or Engraved text). The code uses opencv image filtering techniques to filter the images as clean as possible and then feeds it to Tesseract.  

## Dependencies

  Python 
  OpenCV
  Tesseract
  Numpy
  imutils

### There are three files - 
1. picture_from_webcam.py clicks a single picture from webcam and performs OCR on it
2. read_from_disk.py loads a picture from an image file
3. video_from_webcam.py feeds the realtime video from the webcam (doesn't perform as expected)

### Sample pictures - 
