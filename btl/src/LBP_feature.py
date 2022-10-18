from builtins import range

import numpy as np
import math
import cv2
import os
import math
#lbp 8 x1
# read image  , each image a row
a = np.zeros(9);
a[0] = 2**6;
a[1] = 2**7;
a[2] = 2**0;
a[3] = 2**5;
a[4] = 0;
a[5] =2**1;
a[6] = 2**4;
a[7] = 2**3;
a[8] = 2**2;
index_arr = np.array(
    [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127,
    128, 129,131,135,143,159,191,192,193,195,199,207,223,224,225,227,231,239,240,241,243,247,248,249,251,252,253,254,255]);


def read_data(folder):
   images = []
   for filename in os.listdir(folder):
      img = cv2.imread(os.path.join(folder, filename),0)
      if img is not None:
        images.append(img)
   return images

def lbp_one(img,a,index_arr):
    # img =np.asarray(img)
    r, v = img.shape; # get  shape
    new_img = np.zeros((r + 2, v + 2)); # generate array
    new_img[1:r + 1, 1:v + 1] = img;
    #imgn = np.zeros((r,v))
    his_lbp_one = np.zeros(256);
    for i in range(1, r + 1):
        for j in range(1, v + 1):
            mask = new_img[i - 1:i + 2, j - 1:j + 2] - new_img[i][j];
            mask = np.where(mask >= 0, 0, 1)
            tem = mask.reshape(9) * a;
            index = int(sum(tem))
            his_lbp_one[index] += 1


    t = sum(his_lbp_one)
    #print(imgn)
    his_lbp = np.zeros(59);
    his_lbp[0:58] = his_lbp_one[index_arr]
    his_lbp[58] = t - sum(his_lbp_one[index_arr])
    return (his_lbp)
def lbp(img_arr, a, index_arr):
    his = [] ;
    for i in range(len(img_arr)):
        tem  = lbp_one(img_arr[i],a,index_arr);
        his.append(tem)

    return np.asarray(his);
def distance(hist1, matrix_hist):
    dis = matrix_hist-hist1
    dis *= dis
    dis = sum(dis.T)
    dis = dis **(1/2)
    return dis

# img_arr = read_data("images");
# img_arr =np.asarray(img_arr)
# print(img_arr.shape)
# his = lbp(img_arr,a,index_arr);
# print(his.shape)
test = cv2.imread("C:\\Users\\tiem.nv164039\\Desktop\\images\\00943.png", 0);

his_one =  lbp_one(test,a,index_arr);
print(his_one)
# dis = distance(his_one,his)
# print(dis**(1/2))
# index = np.argsort(dis)[:2]
# cv2.imshow('anh goc',test)
#
# for i in range(len(index)):
#     cv2.imshow('anh'+str(i),img_arr[index[i]]);

cv2.imshow('anh goc', test)

cv2.waitKey(0);
cv2.destroyAllWindows();