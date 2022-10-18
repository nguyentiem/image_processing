import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("ball1.png", 0);
img1 = cv2.imread("ball2.png", 0);

test =  cv2.add(img,img1)
test1 = cv2.subtract(img,-img1)
print(test)
print(test1)
result = cv2.subtract(test,test1)
cv2.imshow('test1',test);
cv2.imshow('test2',test1);
cv2.imshow('result',result);
cv2.waitKey(0);
cv2.destroyAllWindows();