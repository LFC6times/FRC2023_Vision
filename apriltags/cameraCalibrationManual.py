import cv2 as cv
import numpy as np
import glob
import dt_apriltags as apriltag

detector: apriltag.Detector = apriltag.Detector(families="tag16h5", quad_decimate=0, nthreads=4)

detections, img2 = None, None
detect: apriltag.Detection

test_path = '/home/addison/FRC2023_Vision/apriltags/apriltags_test_imgs/*.jpg'
#test_path = "C:/Users/astro/OneDrive/Documents/FRC2023_Vision/apriltags/apriltags_test_imgs/*.jpg"
images = glob.glob(test_path)
print(len(images))

def metersToFeet(meters):
    return meters * 3.2808399

for i, fname in enumerate(images, start=1):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # original: [307, 307, 640, 360]
    detections = detector.detect(gray, True, [307, 307, 640, 360], 0.15244) # tag_size in meters, and does not include the solid border, so 6in -> m is ~0.15244
    detected = 0
    detection: apriltag.Detection
    for idx, detection in enumerate(detections):
        if detection.tag_id < 1 or detection.tag_id > 9 or detection.decision_margin < 20:
            # print("rejected:", detection.tag_id, "dec_mar:", detection.decision_margin)
            continue
        detected += 1
        P = [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]
        pose = P @ detection.pose_R @ (-1 * detection.pose_t)
        print(metersToFeet(pose[0][0]), metersToFeet(pose[1][0]), metersToFeet(pose[2][0]))
        # print("detected:", detection.__str__())
    print("[INFO]", detected, "total AprilTags detected in frame", i)

cv.destroyAllWindows()