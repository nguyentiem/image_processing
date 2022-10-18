#C:\Users\tiem.nv164039\PycharmProjects\DoAnHeNhung\xla\img\bean.jpg
import cv2
import numpy as np
import matplotlib.pyplot as plt

def count(a):
    dem = np.zeros(256);
    for row in range(a.shape[0]): # dung ham python
          for col in range(a.shape[1]):
             dem[a[row][col]]+=1
    dem =dem/(a.shape[0]*a.shape[1])
    return dem;
# tinh tan suat xuat hien muc xam thi k

def phan_phoi(d):
    s =0
    for row in range(d.shape[0]):
        s+=d[row];
        d[row] = s;
    return d*255
# tinh phan phoi tich luy tai muc xam thu k
img = cv2.imread("C:\\Users\\tiem.nv164039\\PycharmProjects\\XuLiAnh\\img\\bean.jpg", 0);
#cv2.imshow('linh',img);
a = np.asarray(img)
d = count(a) # tinh ra xac suat cua moi diem sang
# tinh ham phan phoi xac suat tich luy

plt.hist(img)
plt.plot(d)
d = phan_phoi(d)
for row in range(a.shape[0]):  # dung ham python
    for col in range(a.shape[1]):
        img[row][col] = round(d[img[row][col]])
cv2.imshow('linh1',img);
plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();

# # su dung ham python cho nhanh va ket hop voi nhan ma tran
#
# b1 : tinh xac suat xuat hien cua cac muc sang tu 0-255
# b2 : tinh xac suat tich luy cua cac diem anh ( p(x< muc_sang) )
# B3 : lay xac suat tisch luy nhan voi muc sang toi da (L-1)
# B4 : thay muc sang cac diem anh bang muc sang moi vua tinh
#
#
# #C:\Users\tiem.nv164039\PycharmProjects\DoAnHeNhung\xla\img\bean.jpg
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# def count(a):
#     dem = np.zeros((256,a.shape[2]));
#     for i in range(a.shape[2]):
#        for row in range(a.shape[0]): # dung ham python
#           for col in range(a.shape[1]):
#              dem[a[row][col][i]][i]+=1
#        dem =dem/(a.shape[0]*a.shape[1])
#     return dem;
#
# def phan_phoi(d):
#     for i in range(d.shape[1]):
#         s =0
#         for row in range(d.shape[0]):
#            s+=d[row][i];
#            d[row][i] = s;
#
#     return d*255
#
# def subsitution(a,d):
#      for i in range(a.shape[2]):
#        for row in range(a.shape[0]):  # dung ham python
#            for col in range(a.shape[1]):
#                a[row][col][i] = round(d[a[row][col][i]][i])
#      return a;
#
# img = cv2.imread("C:\\Users\\tiem.nv164039\\PycharmProjects\\DoAnHeNhung\\xla\\img\\xemay.jpg", 0);
# cv2.imshow('linh',img);
# #plt.hist(img)
#
# a = np.asarray(img)
#
# print(a.shape)
#
# d = count(a) # tinh ra xac suat cua moi diem sang
# # tinh ham phan phoi xac suat tich luy
# print(d)
# print(d.shape)
# d = phan_phoi(d)
# a = subsitution(a,d)
#
# cv2.imshow('linh1',img);
# #plt.show()
# cv2.waitKey(0);
# cv2.destroyAllWindows();
#
# # su dung ham python cho nhanh va ket hop voi nhan ma tran
#
# b1 : tinh xac suat xuat hien cua cac muc sang tu 0-255
# b2 : tinh xac suat tich luy cua cac diem anh ( p(x< muc_sang) )
# B3 : lay xac suat tisch luy nhan voi muc sang toi da (L-1)
# B4 : thay muc sang cac diem anh bang muc sang moi vua tinh
