import cv2 as cv
import numpy as np
import util

camMtx = util.readFromFile("newcameramtx.npy")
# camMtxDetectorParam = [camMtx[0][0], camMtx[0][2], camMtx[1][1], camMtx[1][2]]
dist = util.readFromFile("dist.npy")
roi = util.readFromFile("roi.npy")
x, y, w, h = roi

# undistort
mapx, mapy = (util.readFromFile("mapx.npy"), util.readFromFile("mapy.npy"))

cap = cv.VideoCapture(0 + 1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv.CAP_PROP_FPS, 90)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dst = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
    dst = dst[y:y+h, x:x+w]
    cv.imshow("remapped", dst)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()