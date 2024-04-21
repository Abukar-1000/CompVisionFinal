import cv2 as cv
from PIL import Image
from Finger import FingerBase
import numpy as np

class Hand:

    def __init__(self) -> None:
        self.thumb = FingerBase(
            color=[152, 30, 254],
            hsvLowerBoundData = (3,170,250), 
            hsvUpperBoundData = (3,255,255)
        )

        self.index = FingerBase(
            color= [46, 185, 138],
            hsvLowerBoundData=(10, 100, 100),
            hsvUpperBoundData=(10, 255, 255)
        )

        self.middle = FingerBase(
            color= [253, 202, 0],
            hsvLowerBoundData=(5, 170, 200),
            hsvUpperBoundData=(5, 255, 255)
        )

        self.ring = FingerBase(
            color=[250, 92, 162],
            hsvLowerBoundData=(5,110,250),
            hsvUpperBoundData=(5,255,255)
        )

        self.pinky = FingerBase(
            color= [0, 77, 255],
            hsvLowerBoundData=(5,200,150),
            hsvUpperBoundData=(5,255,255)
        )

    def detectFingersOnFrame(self, frame):
        thumbBoundingBox = self.thumb.getBoundingBox(frame)
        indexBoundingBox = self.index.getBoundingBox(frame)
        middleBoundingBox = self.middle.getBoundingBox(frame)
        ringBoundingBox = self.ring.getBoundingBox(frame)
        pinkyBoundingBox = self.pinky.getBoundingBox(frame)

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