import cv2 as cv
import os
import time
import uuid

DATASET_PATH = "dataset"
samplesPerGesture = 15
gestureLabels = [
    "point",
    "click",
    "highlight",
    "scroll_up",
    "scroll_down",
    "view_all_tabs"    
]

for label in gestureLabels:
    os.mkdir(f"{DATASET_PATH}/{label}")
    cap = cv.VideoCapture(1)
    for x in range(samplesPerGesture):
        os.system("clear")
        print(f"collecting for lable: {label}\t{x}x")        
        ret, frame = cap.read()
        imgName = os.path.join(DATASET_PATH, label, label + f".{str(uuid.uuid1())}.jpg")
        cv.imwrite(imgName, frame)
        cv.imshow("frame", frame)

        time.sleep(2)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()