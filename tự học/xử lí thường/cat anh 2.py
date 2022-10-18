from numpy import linalg as la
import numpy as np
import math
import cv2
import os
import math
from skimage.util import random_noise
def read_data(folder):
   images = []
   for filename in os.listdir(folder):
      img = cv2.imread(os.path.join(folder, filename),1)
      if img is not None:
        images.append(img)
   return images
img = read_data("test")

# #img1 = cv2.imread("center2.jpg", 1);
# # C:\Users\tiem.nv164039\Downloads\IMG\left_2020_03_26_16_30_23_449.jpg
# # imgn = img[66:132,50:250]
# # print(imgn.shape)
# #cv2.imwrite("sua_anh.jpg", imgn);
#
# # Add salt-and-pepper noise to the image.
# for i in range(len(img)):
#   # noise_img = random_noise(img[i], mode='s&p', amount=0.03)
#   # noise_img = np.array(255 * noise_img, dtype='uint8')
#   # cv2.imwrite(str(i)+'.png', noise_img)
#   # imgn1 = np.flip(img[i],(1))
#   x1 = np.random.randint(0,128);
#   x2 = np.random.randint(0,128);
#   y1 = np.random.randint(0, 128);
#   y2 = np.random.randint(0, 128);
#   cv2.line(img[i], (x1, y1), (x2, y2), (200, 100, 100), 2)
img = np.zeros((128,128,3),dtype=int)
img[:,:,0]+=255
print(img)
cv2.imwrite('r.png', img)

# imgn=np.zeros((128,128,3),dtype=int)
# imgn[:,:,:]=img
# imgn[:,:,0]-=20
# # imgn[:,:,1]+=20
# # imgn[:,:,2]+=20
# imgn = np.uint8(imgn)
# print(img)
# print(imgn)


cv2.waitKey(0);
cv2.destroyAllWindows();