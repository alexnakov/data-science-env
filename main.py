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
    for detection in out.detections:
      location_data = detection.location_data
      bbox = location_data.relative_bounding_box

      x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

      x1 = int(x1 * W)
      y1 = int(y1 * H)
      w = int(w * W)
      h = int(h * H)

      # cv.rectangle(img, (x1,y1),(x1+w,y1+h),[0,255,255],5)
      img[y1:y1+h,x1:x1+w,:] = cv.blur(img[y1:y1+h,x1:x1+w,:], (27,27))
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

