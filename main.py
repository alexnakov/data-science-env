import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

img_path = os.path.join('.','bear.jpg')
img = cv.imread(img_path)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img.shape)

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

target_display_height = 800

for alpha in range(10, 250, 10):
  ret, threshold_img = cv.threshold(img, alpha, 255, cv.THRESH_BINARY)
  print(alpha)
  display_2_images_side_by_side(img, threshold_img)
