'''
b1  lay histogram tung mau
b2
trung binh E =  1/n * pij
do lech chuan sprt(1/n*(pij-Eij)**2)
doc lech cap 3 :  can bac 3 cua 1/N ((pij-Eij)**3)
'''
from builtins import print
from numpy import linalg as la
import numpy as np
import math
import cv2
import os
import math

# read image  , each image a row
def read_data(folder):
   images = []
   for filename in os.listdir(folder):
      img = cv2.imread(os.path.join(folder, filename),1)
      if img is not None:
        images.append(img)
   return images
# caculate histogram image
def getHis_one(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    color = ('H', 'S', 'V')
    his = [];
    n = (img.shape[0]*img.shape[1])
    for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])  # chanel 0 - 1 -2 cac kenh cua anh
        histr /=n;
        his.append(histr);
    his = np.asarray(his).reshape(3,-1)
    return his

def getHist(matrix):
    his =[]
    for i in range(len(matrix)):
        hist = getHis_one(matrix[i])
        his.append(hist)
    return np.asarray(his)



def getVector_one(hist,N):
#kich thuoc 3*256
    result = []
    index =[]
    index_test = np.arange(256)
    index.append(index_test)
    index.append(index_test)
    index.append(index_test)
    tem =hist*index  # xác suât nhân với giá trị
    E=sum(tem.T).reshape(3,1)     # tính kì vọng
    E/=255
    index = np.asarray(index)
    sub = index/255-E                # i =0 -> 255 - trung binh
    E = E.reshape(-1)
    do_lech = (((sum(((sub**2)*hist).T)))**(1/2)) # do lech chuan
    ttt= sum(((sub**3)*hist).T)
    tem = np.where(ttt<0,-1,1)
    Skewness = ((ttt*tem)**(1/3))*tem
    # print(E.shape)
    result.append(E)
    # print(E)
    # print(do_lech.shape)
    result.append(do_lech)
    # print(Skewness.shape)
    result.append(Skewness)
    return np.asarray(result).T

def getVetor(hist,N):
    n = len(hist);
    vec=[]
    for i in range(n):
        his = getVector_one(hist[i],N)
        vec.append(his)
    return np.asarray(vec)


def distance(hist1, matrix_hist):
    dis = matrix_hist-hist1

    dis *= dis
    return sum(dis.T)

# img_matrix = read_data("images");

test = cv2.imread("C:\\Users\\tiem.nv164039\\Desktop\\images\\00943.png", 1);
r,v,h =test.shape
N = r*v
#img_matrix = np.asarray(img_matrix)
# hist = getHist(img_matrix)
hist1 = getHis_one(test)
E1 = getVector_one(hist1,N)
print(E1)
# E2 = getVetor(hist,N)
# distan = distance(E1,E2)
# print(distan)
# index = np.argsort(distan)[:2]
# test = cv2.cvtColor(test, cv2.COLOR_BGR2HSV);
cv2.imshow('anh goc',test)
# for i in range(len(index)):
#     # img_matrix[index[i]] = cv2.cvtColor(img_matrix[index[i]], cv2.COLOR_BGR2HSV);
#     cv2.imshow('anh'+str(i),img_matrix[index[i]]);

cv2.waitKey(0);
cv2.destroyAllWindows();

