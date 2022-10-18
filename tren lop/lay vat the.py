import cv2
import numpy as np
img = cv2.imread("C:\\Users\\tiem.nv164039\\PycharmProjects\\XuLiAnh\\img\\test.png", 0);
img1 = cv2.imread("C:\\Users\\tiem.nv164039\\PycharmProjects\\XuLiAnh\\img\\test1.png", 0);
'''
I1  = 255 - img

I2 =  255 - img1
I3 = I1+I2
# a = I1 - I2 # co ca am duong

a= cv2.subtract(I1,I2) # chi giu lai duong

for  i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if a[i][j] ==-255 :
            print(i,j)
a = np.uint8(a)
#cv2.imshow('a1',a);
I5 = I3 -a
I6 = I1-I5
I7  = I2 - I5
'''
#cv2.imshow('anh1',img);
#cv2.imshow('anh2',img1);
# convert to int hoac dung cach khac 

i12 = img - img1   # tu convert ve uint8   chuyen goc khi tru
i21 = img1 - img  # tu convert ve uint8



tem = np.where(i12!=0,255,0)
tem = tem.astype(np.uint8)

test = cv2.bitwise_and(img,tem)
test1 = cv2.bitwise_and(img1,tem)
T  = img+img1       # chuyen goc khi cong
#T = np.where(tem==0,0,T)

# for row in range(T.shape[0]):
#     for col in range(T.shape[1]):
#         print(T[row][col])

#cv2.imshow('tong',T);
re = T - abs(i12)
re = re /2
result = test - re
#cv2.imshow('re',result);




cv2.waitKey(0);
cv2.destroyAllWindows();

