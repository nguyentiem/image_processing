import cv2
import numpy as np
img = cv2.imread("test.png", 0);
img = np.asarray(img)
img = np.where(img>235,255,0);
img = np.uint8(img)
cv2.imshow('linh',img);
cv2.waitKey(0);
cv2.destroyAllWindows();
