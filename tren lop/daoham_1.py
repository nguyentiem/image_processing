from builtins import print

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread("1.png",1)
dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) # ảnh sau đạo hàm theo trục x
dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) # ảnh sau đạo hàm theo trục y
laplacian = cv2.Laplacian(img, cv2.CV_64F) # ảnh sau khi đạo hàm cấp 2
dxy =  np.sqrt(np.square(dx) + np.square(dy))
dxy = np.uint8(dxy)
cv2.imshow('anh goc',img)
cv2.imshow('dx',dx)
cv2.imshow('dy',dy)
cv2.imshow('dxy',dxy)
cv2.waitKey(0);
cv2.destroyAllWindows();