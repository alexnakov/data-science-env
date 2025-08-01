import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

def display_2_images_side_by_side(img1, img2, target_height=400):
  aspect_ratio1 = img1.shape[1] / img1.shape[0]
  aspect_ratio2 = img2.shape[1] / img2.shape[0]

  img1_resized = cv.resize(img1, (int(target_height * aspect_ratio1), target_height))
  img2_resized = cv.resize(img2, (int(target_height * aspect_ratio2), target_height))

  cv.namedWindow('img1', cv.WINDOW_AUTOSIZE)
  cv.imshow('img1',img1_resized )
  cv.moveWindow('img1', 100, 100)

  offset_x = 100 + img1_resized.shape[1] + 20
  cv.namedWindow('img2', cv.WINDOW_AUTOSIZE)
  cv.imshow('img2', img2_resized)
  cv.moveWindow('img2',offset_x, 100)

  cv.waitKey(0)
  cv.destroyAllWindows()

img1 = cv.imread(os.path.join('./assets','birds.jpeg'))
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
ret, img2 = cv.threshold(img1, 127, 255, cv.THRESH_BINARY_INV)
contours, hierarchy = cv.findContours(img2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

color = (0,255,0)
for cnt in contours:
  if cv.contourArea(cnt) > 200:
    # cv.drawContours(img1, cnt, -1, color, 1)

    x1,y1,w,h=cv.boundingRect(cnt)
    cv.rectangle(img1, (x1,y1),(x1+w,y1+h),color,3)

display_2_images_side_by_side(img1, img2)
