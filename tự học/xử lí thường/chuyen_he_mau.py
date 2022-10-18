import cv2
import numpy as np
img = cv2.imread("images/a1.jpg", 1);
new = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);
#img2 = cv2.add(img,-50)
cv2.imshow('tiem',new);
cv2.waitKey(0);
cv2.destroyAllWindows();