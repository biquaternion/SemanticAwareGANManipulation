#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import cv2
import numpy as np
from copy import deepcopy


def overlay_image(foreground_image, background_image, mask):
    background_mask = cv2.bitwise_not(mask)
    foreground_mask = cv2.blur(mask, (7, 7))
    #     background_mask = cv2.bitwise_not(foreground_mask)
    masked_fg = (foreground_image * (1 / 255.0)) * (np.tile(foreground_mask[:, :, None], [1, 1, 3]) * (1 / 255.0))
    masked_bg = (background_image * (1 / 255.0)) * (np.tile(background_mask[:, :, None], [1, 1, 3]) * (1 / 255.0))
    out = cv2.addWeighted(masked_fg, 255.0, masked_bg, 255.0, 0.0)
    return np.uint8(out)


if __name__ == '__main__':
    skin_mask = cv2.imread('skin_color_correction/skin_color_correction/sample_img/mask/refined_mask.png')
    body_img = cv2.imread('skin_color_correction/skin_color_correction/sample_img/body/body.jpg')

    skin_mask_erode = cv2.erode(skin_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    fg_original = cv2.bitwise_and(skin_mask_erode, body_img)
    fg = deepcopy(fg_original).astype(float)
    bg = cv2.bitwise_and(255 - skin_mask, body_img)
    fg[:, :, 0] *= 0.53
    fg[:, :, 1] *= 0.62
    fg[:, :, 2] *= 0.82
    fg = fg.astype(np.uint8)

    cv2.imshow('fg', fg)
    cv2.imshow('fg_original', fg_original)
    diff = fg / 255.0 - fg_original / 255
    diff = cv2.blur(diff, ksize=(2, 2))
    print(diff.dtype)
    patch = diff * 255.0
    cv2.imshow('dfg', (body_img + patch).astype(np.uint8))
    cv2.waitKey()