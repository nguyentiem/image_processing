import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, convolve

def my_convolve2d(a, conv_filter):
    submatrices = np.array([
         [a[:-2,:-2], a[:-2,1:-1], a[:-2,2:]],
         [a[1:-1,:-2], a[1:-1,1:-1], a[1:-1,2:]],
         [a[2:,:-2], a[2:,1:-1], a[2:,2:]]])
    multiplied_subs = np.einsum('ij,ijkl->ijkl',conv_filter,submatrices)
    return np.sum(np.sum(multiplied_subs, axis = -3), axis = -3)
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

# img = cv2.imread("C:\\Users\\tiem.nv164039\\PycharmProjects\\XuLiAnh\\img\\bean.jpg", 0);
# co = np.array([[0,1,0],[1,-4,1],[0,1,0]]);
# co = np.uint8(co)
# print(img.shape)
# print(co.shape)
# imgn= my_convolve2d(img,co)
# imgn = np.uint8(imgn)
# cv2.imshow('linh1',img);
# cv2.imshow('linh',imgn);
a = np.random.randint(0,10,(5,5))
print(a)
matrix  = np.array([[0,1,0],[1,-4,1],[0,1,0]]);

b = cov(a,matrix)
print(b)
plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();