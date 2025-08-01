import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from utils import *
from PIL import Image

YELLOW = [0,255,255]
video_capture = cv.VideoCapture(0)
yellow_lower_limit, yellow_upper_limit = get_color_limits(YELLOW, tol=15)

while True:
  ret, frame = video_capture.read()

  hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
  mask = cv.inRange(hsv_img, yellow_lower_limit, yellow_upper_limit)
  mask_ = Image.fromarray(mask)

  bbox = mask_.getbbox()

  if bbox is not None:
    x1,y1,x2,y2 = bbox
    frame = cv.rectangle(frame, (x1,y1),(x2,y2),[0,255,0],2)

  cv.imshow('frame', frame)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv.destroyAllWindows()
