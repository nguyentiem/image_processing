import numpy as np
import cv2
import pandas as pd
import os

# data_dir = "C:\\Users\\tiem.nv164039\\Desktop\\"
#
# data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'link.csv'), names=['img','idol'])
# X = data_df['img'].values
# Y = data_df['idol'].values;

test1 = cv2.imread("../images/tt.jpg", 1)

test = np.asarray(test1)
r,c,d = test.shape
cr = 78
cc = int(c/2)
test = test[cr-78:cr+78,cc-90:cc+90,:]



# x = []
# y=[]
#
# for i in range(len(X)):
#     img  = cv2.imread(X[i],1)
#     img = np.asarray(img)
#     x.append(img)
#     y.append(Y[i])
#     imgn1 = np.flip(img, (1))
#     x.append(imgn1)
#     y.append(Y[i])
# x = np.asarray(x)
# y = np.asarray(y)