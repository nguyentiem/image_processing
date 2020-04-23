from builtins import print
from numpy import linalg as la
import numpy as np
import math
import cv2
import os


N =20
# read data
def read_data(folder):
   images = []
   for filename in os.listdir(folder):
      img = cv2.imread(os.path.join(folder, filename),0)
      if img is not None:
        img=  np.asarray(img).reshape(-1)
        images.append(img)
   return images


# caculation new matrix from average vector
def caculation(a):
    b=[sum(x)/a.shape[0] for x in zip(*a)]
    #np.asarray(list(map(sum, zip(*a))))
    b = np.asarray(b).reshape(-1)
    for i in range(a.shape[0]):
        a[i] = a[i]-b
    return a,b

# caculate covarience matrix
def cov_matrix(a):
    C  = (np.dot(a.T,a)) /(a.shape[0]-1) # (16k x 10 )*(10 x 16k) => 16kx16k  /(a.shape[0]-1)
    D  = np.cov(a.T)
    return C,D

# caculate eigenvalues and eigvector
# chose 20 vector
def lamda_vector(C):
    w,v = la.eig(C);
    # chon ra 20 vector ung voi tri rieng lon nhat
    # duyet tu dau den cuoi lay ra cac cot lon nhat
    index  = sorted(range(len(w)), key = lambda sub: w[sub])[-N:]
    V=[]
    for i in index:
        tem = v[:,i] # ma tran hang
        V.append(tem)
    return  V

# get vector
def get_vector(X):
    a, b = caculation(X)
    C, D = cov_matrix(a)  # 16k*16k
    np.savetxt("cov_matrix.txt", C, fmt="%6f")
    V = np.asarray(lamda_vector(C)).transpose()  # V 16k*20
    np.savetxt("pca_vector.txt", V, fmt="%6f")
    return V

# chuyen khong gina cho vector truc giao voi nhau bieu dien ma tran anh
# caculation Y form X
# Y = X * v
def convert_PCA(X,V):
    # y = np.loadtxt("test.txt")
    Y = []
    for i in range(X.shape[0]):
        Y.append(np.dot(X[i], V))
    Y = np.asarray(Y)
    return Y

# caculate distance vector and matrix
def distance(x,y):
    dis = []
    for i in range(x.shape[0]):
        dis.append(math.sqrt(sum([(a - b) ** 2 for a, b in zip(x[i], y)]))/x.shape[0])
    return dis

X = np.asarray(read_data("images"))   #  n*16k
np.savetxt("matrix_data.txt", X, fmt="%6f")
V = get_vector(X)
Y = convert_PCA(X,V)
np.savetxt("pca_output.txt",Y, fmt="%6f")

test = cv2.imread("test_sample/69024.png", 0);
test = np.asarray(test).reshape(-1);

test = np.dot(test,V)
dis = distance(Y,test)

print(dis)
print(len(dis))
np.savetxt("distance.txt", dis, fmt="%6f")





