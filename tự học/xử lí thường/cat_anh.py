import cv2
import numpy as np
img = cv2.imread("ac.jpg", 1);
cv2.line(img,(100,100),(400,400),(200,100,100),5)
cv2.imwrite("sua_anh.jpg", img);
#cv2.imshow('tiem',img);
#cv2.waitKey(0);
cv2.destroyAllWindows();