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

def co_occurrence(img):
    r,c = img.shape
    matrix = np.zeros((255,255));
    for i in range(r-1):
        for j in range(c-1):
            matrix[img[i][j]][img[i+1][j+1]]+=1;

    return matrix;
a =np.random.randint(0,3,(3,3))
print(a)
max  =co_occurrence(a)
# laays his ma tran dong hien .

print(max)