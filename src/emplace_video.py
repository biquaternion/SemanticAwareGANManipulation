#!/usr/bin/env python3
#-*- coding utf-8 -*-
from pathlib import Path

import numpy as np

from emplace_face import align_faces
from mtcnn import mtcnn
import cv2

if __name__ == '__main__':
    detector = mtcnn.MTCNN()
    body_im = cv2.imread('data/body/body_02.png')
    faces = detector.detect_faces(body_im)
    face = faces[0]
    bd_pts = np.array([[x, y] for k, (x, y) in face['keypoints'].items()])
    cap = cv2.VideoCapture(0)
    stop = False
    while not stop:
        ok, im = cap.read()
        if ok:
            faces = detector.detect_faces(im)
            if faces:
                face = faces[0]
                bx, by, bw, bh = face['box']
                pts = np.array([[x - bx, y - by] for k, (x, y) in face['keypoints'].items()])
                aligned = align_faces(face_im=im[by:by+bh, bx:bx+bw, :], body_im=body_im, face_pts=pts, body_pts=bd_pts)
                cv2.imshow('aligned', aligned['result_pers'])
                key = cv2.waitKey(1)
                if key == ord('q'):
                    stop = True