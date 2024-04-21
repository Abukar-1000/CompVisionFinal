import cv2 as cv
from PIL import Image
import numpy as np

class FingerBase:

    def __init__(self, color, hsvLowerBoundData, hsvUpperBoundData) -> None:
        self.color = color
        self.lowerBound = hsvLowerBoundData
        self.upperBound = hsvUpperBoundData

    def __getHsvInterval(self):
        color = np.uint8([[self.color]])
        hsvColor = cv.cvtColor(color, cv.COLOR_BGR2HSV)
        
        lowerBound = hsvColor[0][0][0] - self.lowerBound[0], self.lowerBound[1], self.lowerBound[2]
        upperBound = hsvColor[0][0][0] + self.upperBound[0], self.upperBound[1], self.upperBound[2]
        lowerBound = np.array(lowerBound, dtype=np.uint8)
        upperBound = np.array(upperBound, dtype=np.uint8)

        return lowerBound, upperBound
    
    def getBoundingBox(self, frame):
        hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lowerBound, upperBound = self.__getHsvInterval()
        mask = cv.inRange(hsvFrame, lowerBound, upperBound)
        maskI = Image.fromarray(mask)
        boundB = maskI.getbbox()
        return boundB