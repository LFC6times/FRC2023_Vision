import numpy as np
import cv2 as cv

img = cv.imread("test.jpg")

src = cv.cuda_GpuMat()
dst = cv.cuda_GpuMat()

src.upload(img)

print(type(src), type(dst), type(img))
cv.cuda.cvtColor(src, cv.COLOR_BGR2GRAY, dst)
print(src.type(), cv.CV_32FC2)

# dst = cv.cuda_GpuMat(src.size(), cv.CV_32FC1)
# dst = cv.cuda.magnitude(src, dst)
 
result = dst.download()
 
cv.imshow("result", result)
cv.waitKey(0)