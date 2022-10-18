import cv2
import numpy as np

def ham(L, r1,s1,r2,s2):
    for row in range(L.shape[0]):
        for col in range(L.shape[1]):
            if L[row][col] < r1 :
                L[row][col] = (L[row][col]/r1)*s1
            elif  L[row][col]<r2:
                L[row][col]=  ((L[row][col]-r1)/(r2-r1))*(s2-s1)+s1
            else:
                L[row][col] = ( (L[row][col]-r2)/(255-r2))*(255-s2)+s2
    return L

img = cv2.imread("test.png", 0);

cv2.imshow('linh',img);

print(img)
cv2.waitKey(0);
cv2.destroyAllWindows();
