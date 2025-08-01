import cv2 as cv
import numpy as np

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

def get_color_limits(color, tol=10):
  c = np.uint8([[color]])
  hsv_color = cv.cvtColor(c, cv.COLOR_BGR2HSV)

  lower_limit = hsv_color[0][0][0] - tol, 100, 100
  upper_limit = hsv_color[0][0][0] + tol, 255, 255

  lower_limit = np.array([lower_limit], dtype=np.uint8)
  upper_limit = np.array([upper_limit], dtype=np.uint8)

  return lower_limit, upper_limit

