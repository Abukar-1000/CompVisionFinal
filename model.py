import cv2 as cv
from PIL import Image
import numpy as np
from Hand import Hand

def getHsvInterval(bgrColor):
    bgrColor = np.uint8([[bgrColor]])
    hsvColor = cv.cvtColor(bgrColor, cv.COLOR_BGR2HSV)
    
    lowerBound = hsvColor[0][0][0] - 5, 110, 250
    upperBound = hsvColor[0][0][0] + 5, 255, 255
    lowerBound = np.array(lowerBound, dtype=np.uint8)
    upperBound = np.array(upperBound, dtype=np.uint8)

    return lowerBound, upperBound

def getBoundingBox(frame, color):
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lowerBound, upperBound = getHsvInterval(color)
    mask = cv.inRange(hsvFrame, lowerBound, upperBound)
    maskI = Image.fromarray(mask)
    boundB = maskI.getbbox()
    return boundB

def detectHand(frame):
    thumbColor = [152, 30, 254]
    indexColor = [46, 185, 138]
    middleColor = [253, 202, 0]
    ringColor = [250, 92, 162]
    pinkyColor = [0, 77, 255]

    thumbBoundingBox = getBoundingBox(frame, thumbColor)
    indexBoundingBox = getBoundingBox(frame, indexColor)
    middleBoundingBox = getBoundingBox(frame, middleColor)
    ringBoundingBox = getBoundingBox(frame, ringColor)
    pinkyBoundingBox = getBoundingBox(frame, pinkyColor)

    if thumbBoundingBox:
        x1, y1, x2, y2 = thumbBoundingBox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    
    if indexBoundingBox:
        x1, y1, x2, y2 = indexBoundingBox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
    
    if middleBoundingBox:
        x1, y1, x2, y2 = middleBoundingBox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 4)
    
    if ringBoundingBox:
        x1, y1, x2, y2 = ringBoundingBox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
    
    if pinkyBoundingBox:
        x1, y1, x2, y2 = pinkyBoundingBox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
    
    return frame

handObj = Hand()
cap = cv.VideoCapture(1)
while True:
    ret, frame = cap.read()

    frame = handObj.detectFingersOnFrame(frame)
    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()