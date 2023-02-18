import dearpygui.dearpygui as dpg
import util
import cv2 as cv
import numpy as np

camMtx = util.readFromFile("newcameramtx.npy")
# camMtxDetectorParam = [camMtx[0][0], camMtx[0][2], camMtx[1][1], camMtx[1][2]]
dist = util.readFromFile("dist.npy")
roi = util.readFromFile("roi.npy")
x, y, w, h = roi

# undistort
mapx, mapy = (util.readFromFile("mapx.npy"), util.readFromFile("mapy.npy"))

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv.CAP_PROP_FPS, 90)

fx, fy, cx, cy = 0, 0, 0, 0
oldfx, oldfy, oldcx, oldc = fx, fy, cx, cy

def main():
    dpg.create_context()
    #create viewport
    dpg.create_viewport(title='Team 3952', width=720, height=480)
    dpg.set_viewport_vsync(True)
    dpg.setup_dearpygui()
    dpg.set_global_font_scale(3)

    with dpg.window(tag="Window1"):
        dpg.set_primary_window("Window1", True)
    
    with dpg.window(tag="CalibWin"):
        dpg.add_input_text(label="fx", tag="fx")
        dpg.add_input_text(label="fy", tag="fy")
        dpg.add_input_text(label="cx", tag="cx")
        dpg.add_input_text(label="cy", tag="cy")
        
main()

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