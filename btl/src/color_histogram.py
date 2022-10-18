
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

     color = ('r', 'g', 'b')
     his = [];
     n = (img.shape[0]*img.shape[1])
     for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])  # chanel 0 - 1 -2 cac kenh cua anh
        histr/=n# 0-1
        his.append(histr);
        # plt.plot(histr, color=col)
        # plt.xlim([0, 256])
     his = np.asarray(his).reshape(-1)
     # plt.show()
     return his


def getHist(matrix):
    his =[]
    for i in range(len(matrix)):
        hist = getHis_one(matrix[i])
        his.append(hist)
    return np.asarray(his)



def distance_eu(hist1, matrix_hist):
    dis = matrix_hist-hist1
    dis *= dis
    dis  = sum(dis.T)
    dis = dis**(1/2)
    return dis

def distance_cos(hist1,matrix_hist):

    dis = matrix_hist*hist1
    size_1 = (sum(hist1*hist1))**(1/2)# 1

    size_v = matrix_hist*matrix_hist
    size_v = sum(size_v.T)
    size_v = size_v**(1/2)

    dis =sum(dis.T)
    dis = dis/(size_v*size_1);
    return dis

# img_matrix = read_data("images");
img = cv2.imread("C:\\Users\\tiem.nv164039\\Desktop\\images\\00943.png", 1);

# hist = getHist(img_matrix)
hist1=getHis_one(img)
print(max(hist1))
print(min(hist1))
# distan = distance_cos(hist1,hist)
# plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();
