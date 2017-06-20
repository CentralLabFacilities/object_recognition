import cv2
import numpy as np
import sys
import os
from fractions import Fraction


def change_background(fg, bg, mask):
        mask_inv = cv2.bitwise_not(mask)
	fg = cv2.bitwise_and(fg, fg, mask = mask)
	bg = cv2.bitwise_and(bg, bg, mask = mask_inv)
	cv2.imwrite('fg.jpg',fg)
	cv2.imwrite('bg.jpg',bg)
	res = fg + bg
	cv2.imshow('res', res)
	cv2.imwrite('res.jpg',res)


if __name__ == "__main__":

    fg = cv2.imread('biscuits.jpg',1)
    bg = cv2.imread('rgb-0.ppm',1)
    mask = cv2.imread('biscuits_mask.jpg',0)
    change_background(fg, bg, mask)

    print '\033[1m\033[92mDone!\033[0m'
