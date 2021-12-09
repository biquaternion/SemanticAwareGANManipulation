# /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from argparse import ArgumentParser

from mtcnn import mtcnn

SUPPORTED_EXTS = ['.jpg', '.png']
DELAY_MS = 100


def align_faces(body_im, face_im, body_pts, face_pts):
    homo, _ = cv2.findHomography(face_pts[[0, 1, 2, 3, 4]], body_pts[[0, 1, 2, 3, 4]])
    # thr, _ = cv2.threshold(np.max(im2, axis=2), 254, 255, cv2.THRESH_TOZERO_INV)
    # thr, im2[:, :, 0] = cv2.threshold(im2[:, :, 0], thr, 255, cv2.THRESH_TOZERO_INV)
    # thr, im2[:, :, 1] = cv2.threshold(im2[:, :, 1], thr, 255, cv2.THRESH_TOZERO_INV)
    # thr, im2[:, :, 2] = cv2.threshold(im2[:, :, 2], thr, 255, cv2.THRESH_TOZERO_INV)
    thr, im_aff = cv2.threshold(face_im, 254, 255, cv2.THRESH_TOZERO_INV)
    thr, im_pers = cv2.threshold(face_im, 254, 255, cv2.THRESH_TOZERO_INV)
    aff = cv2.getAffineTransform(face_pts[:3].astype(np.float32), body_pts[:3].astype(np.float32))
    warped_aff = cv2.warpAffine(im_aff, aff, dsize=(body_im.shape[1], body_im.shape[0]))
    warped_pers = cv2.warpPerspective(im_pers, homo, dsize=(body_im.shape[1], body_im.shape[0]))
    thr, mask_aff = cv2.threshold(warped_aff[:, :, 0],
                                  0, 255,
                                  cv2.THRESH_BINARY_INV)
    mask_aff_diff = cv2.dilate(mask_aff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) - mask_aff
    # warped_aff = cv2.GaussianBlur(warped_aff, (3, 3), 3)
    # warped_aff = cv2.GaussianBlur(warped_aff, (3, 3), 3)
    thr, mask_pers = cv2.threshold(warped_pers[:, :, 0],
                                   0, 255,
                                   cv2.THRESH_BINARY_INV)
    canvas_aff = cv2.bitwise_or(warped_aff, body_im, mask=mask_aff)
    canvas_pers = cv2.bitwise_or(warped_pers, body_im, mask=mask_pers)
    result_aff = cv2.bitwise_or(canvas_aff, warped_aff)
    result_pers = cv2.bitwise_or(canvas_pers, warped_pers)
    result = {'warped_aff': warped_aff,
              'warped_pers': warped_pers,
              'mask_aff': mask_aff,
              'mask_pers': mask_pers,
              'result_aff': result_aff,
              'result_pers': result_pers}
    return result


def diff_masks(im1, im2, pts1, pts2):
    thr, im_aff = cv2.threshold(im2, 254, 255, cv2.THRESH_TOZERO_INV)
    aff = cv2.getAffineTransform(pts1[:3].astype(np.float32), pts2[:3].astype(np.float32))
    warped_aff = cv2.warpAffine(im_aff, aff, dsize=(im1.shape[1], im1.shape[0]))


def list_from_path(src_path: Path):
    if src_path.is_dir():
        return [f for f in src_path.glob('*') if f.suffix in SUPPORTED_EXTS]
    else:
        if src_path.suffix in SUPPORTED_EXTS:
            return [src_path]
    return None


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--body', required=True, default='body.png')

    argparser.add_argument('--face', required=True, default='face.png')

    argparser.add_argument('--dst', required=False, default='output')

    argparser.add_argument('--output_fields', required=False, action='append')

    args = argparser.parse_args()
    body_path = Path(args.body)
    face_path = Path(args.face)
    dst_path = Path(args.dst)
    output_fields = args.output_fields

    if not body_path.exists() or not face_path.exists():
        print(f'one of body path : {body_path} or face path : {face_path} does not exist')
        exit(-1)

    dst_path.mkdir(exist_ok=True, parents=True)

    face_list = list_from_path(face_path)
    body_list = list_from_path(body_path)

    results = []
    for face, body in product(face_list, body_list):
        fc_im = cv2.imread(str(face), -1)
        if fc_im.shape[-1] == 4:
            trans_mask = fc_im[:, :, 3] == 0
            fc_im[trans_mask] = [255, 255, 255, 255]
            fc_im = cv2.cvtColor(fc_im, cv2.COLOR_BGRA2BGR)
        bd_im = cv2.imread(str(body))
        detector = mtcnn.MTCNN()
        face_detections = detector.detect_faces(fc_im)
        body_detections = detector.detect_faces(bd_im)
        fc_pts = np.array([[x, y] for k, (x, y) in face_detections[0]['keypoints'].items()])
        bd_pts = np.array([[x, y] for k, (x, y) in body_detections[0]['keypoints'].items()])
        results.append((face, body, align_faces(bd_im, fc_im, bd_pts, fc_pts)))
    for face, body, result in results:
        if output_fields:
            result = {k for k in result if k in output_fields}
        for subj, output_im in result.items():
            cv2.imshow(subj, output_im)
            cv2.waitKey(DELAY_MS)
            output_name = (dst_path / f'{face.stem}_{body.stem}_{subj}.png')
            cv2.imwrite(str(output_name), output_im)
    cv2.waitKey()
