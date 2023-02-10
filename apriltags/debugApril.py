import apriltag
import numpy as np
import cv2 as cv

options: apriltag.DetectorOptions = apriltag.DetectorOptions(families="tag16h5", debug=1)
detector: apriltag.Detector = apriltag.Detector(options)

imagePath = "test.jpg"
image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)

detections, img = detector.detect(image, True)
print(len(detections))

while True:
    cv.imshow("det", img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cv.destroyAllWindows()