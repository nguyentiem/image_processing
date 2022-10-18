import cv2
import numpy as np
img = cv2.imread("images/a1.jpg", 1);
img1 = cv2.imread("images/a2.jpg", 1);
img2 = cv2.addWeighted(img,0.6,img1,0.4,0) # khong trong so thif dung cv2.add(img,img1)
img2 = np.uint8(img2)
cv2.imshow('tiem',img2);
cv2.waitKey(0);
cv2.destroyAllWindows();