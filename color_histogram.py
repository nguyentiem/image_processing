'''
b1  lay histogram tung mau
b2
'''
from distutils.cmd import install_misc

import cv2
import numpy as np
import matplotlib.pyplot as plt
# caculate histogram image
def getHis(img):
    color = ('b', 'g', 'r')
    his = [];
    n = (img.shape[0]*img.shape[1])
    for channel, col in enumerate(color):

        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])  # chanel 0 - 1 -2 cac kenh cua anh
        histr/=n
        his.append(histr);
    his = np.asarray(his).reshape(-1)
    return his
# convolution image
def cov(a,matrix):
    n = matrix.shape[0];
    h = a.shape[0];
    w = a.shape[1]
    b = np.zeros((h+2*(n-2),w+2*(n-2)))
    b[n-2:h+n-2,n-2:w+n-2] =a;
    re = np.zeros((h,w));
    for i in range(n-2,h+n-2):
        for j in range(n - 2, w + n - 2):
            print((matrix*b[i-(n-2):i+(n-2)+1,j-(n-2):j+(n-2)+1]))
            re[i-(n-2)][j-(n-2)] = sum(sum(matrix*b[i-(n-2):i+(n-2)+1,j-(n-2):j+(n-2)+1]))
    print(re)
    return re;

def distance(hist1, matrix_hist):
    dis = matrix_hist-hist1

    dis *= dis
    return sum(dis.T)



#img = cv2.imread("images/00000.png", 1);
img = np.zeros((128,128,3))
img[1:32,1:32,1] =255
img1 = np.zeros((128,128,3))
img1[33:64,33:64,1] =255
img = np.uint8(img)
img1 = np.uint8(img1)

hist1 = getHis(img)
hist2 = getHis(img1)
print(hist1.shape)
print(hist2.shape)
dis = distance(hist1,hist2)
print(dis)


#plt.title('Histogram for color scale picture')
cv2.imshow('image',img);
cv2.imshow('image1',img1);
plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();
