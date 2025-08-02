import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from utils import *
import mediapipe as mp
import argparse

args = argparse.ArgumentParser()
args.add_argument("--mode", default='video')
args.add_argument("--filePath", default='./assets/person.jpg')
parsed_args = args.parse_args()


def process_img(img, face_detection):
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  out = face_detection.process(img)
  H, W, _ = img.shape

  if out.detections is not None:
    for detection in out.detections: # if more the one face
      location_data = detection.location_data
      bbox = location_data.relative_bounding_box

      x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

      x1 = int(x1 * W)
      y1 = int(y1 * H)
      w = int(w * W)
      h = int(h * H)

      center = (x1 + w // 2, y1)
      r = w // 2
      # cv.circle(img, center, r, 255, -1)
      # cv.rectangle(img, (x1,y1),(x1+w,y1+h),[0,255,255],5)

      mask = np.zeros(img.shape[:2], dtype=np.uint8)

      cv.circle(mask, center, r, 255, -1) # This creates the mask you want w/ specific colours
      cv.rectangle(mask, (x1,y1), (x1+w,y1+h), 255, -1)

      blurred_whole_img = cv.blur(img, (51,51))
      if len(img.shape) == 3:
        mask_3ch = cv.merge([mask]*3)
      else:
        mask_3ch = mask
      img = np.where(mask_3ch == 255, blurred_whole_img, img)
    
  img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
  return img

img_path = os.path.join('./assets/','person.jpg')
img = cv.imread(img_path)

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
  if parsed_args.mode in ['image']:
    img = cv.imread(parsed_args.filePath)
    output_img = process_img(img, face_detection)
    cv.imwrite(os.path.join('./assets/','outputPerson.jpg'), output_img)
  elif parsed_args.mode in ['video']:
    video_capture = cv.VideoCapture(0)
    while True:
      ret, frame = video_capture.read()
      processed_frame = process_img(frame, face_detection)
      cv.imshow('img', processed_frame)
      if cv.waitKey(40) & 0xFF == ord('q'):
        break

