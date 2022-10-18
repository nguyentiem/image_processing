from builtins import print

import numpy as np
import cv2
from matplotlib import pyplot as plt
# cv2.imshow("fft1",np.uint8(magnitude_spectrum))

#tinh pho bien do thu cong
# re = np.real(fft)
# im = np.imag(fft)
# result = 20 * np.log(cv2.magnitude(re, im))
# re1 = (re*re+im*im)**(1/2)
# #print(re1.shape)
# # print(max(re1))
# re1 = 20 * np.log(re1)
# # print(re1)
# re1 = np.uint8(re1)
#re2 = np.uint8(abs(fft))# chinh la pho bien do theo li thuyet re1 =re2
#print(sum(sum(re1 - re2))) == 0
# cv2.imshow("fft",np.uint8(result))
# cv2.imshow("fft1",re1)
# Circular HPF mask, center circle is 0, remaining all ones


# ham numpy
def loc_thong_thap(img):
    rows, cols = img.shape
    # print(str(rows)+" "+str(cols))
    crow, ccol = int(rows / 2), int(cols / 2)
    # print(str(crow)+" "+str(ccol))
    mask = np.ones((rows, cols, 2), np.uint8)

    r = 80
    center = [crow, ccol]  # list gom 2 phan tu
    print(center)
    x, y = np.ogrid[:rows, :cols]

    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    return ;
def loc_thong_cao(img):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 100
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1
    return
def loc_thong_dai(img):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    r_out = 80
    r_in = 10
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                               ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1




img = cv2.imread('1.png', 0)
r,c = img.shape
cv2.imshow("anh goc",img)

fft  = np.fft.fft2(img)
fft = np.fft.fftshift(fft)

#loc thong cao
mask = np.zeros((r, c), np.uint8)
n =15
a = np.arange(r).reshape(r,1); # tạo ma trận theo hàng
b = np.arange(c).reshape(1,c) # tạo ma trận theo cột
M = int(r/2)    #lấy tâm
N = int(c/2)     # lây tâm
a =(a-M)**(2);
b=(b-N)**2;
d = (a+b)**(1/2)
H = np.where(d<n,0,1)
mask = H
fft = fft*mask
ifft = np.fft.ifft2(fft)
imgn = np.uint8(abs(ifft))
imgc = np.uint8(np.where(imgn>30,255,0))
print(imgc)
print(imgn)
cv2.imshow("loc",imgn)
cv2.imshow("l",imgc)
# cv2.imshow("fft1",re1)

cv2.waitKey(0);
cv2.destroyAllWindows();