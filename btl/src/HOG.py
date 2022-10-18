'''
 g = sqrt { g^2_x + g^2_y } theta = arctan c{g_y}{g_x}
 

Tiền xử lý
Tính gradient  (Đối với hình ảnh màu, gradient của ba kênh(red, green và blue) được đánh giá.
                   Độ lớn của gradient tại một điểm ảnh là
                   giá trị lớn nhất của cường độ gradient của ba kênh,
                   và góc là góc tương ứng với gradient tối đa.)
                   góc từ 0-180
Tính vector đặc trưng cho từng ô (cells) : histogram cua cell theo độ lớn mag

Chuẩn hóa khối (blocks)
Tính toán vector HOG
 '''

import cv2
import os
import numpy as np
from numpy import maximum
from skimage  import feature
def read_data(folder):
   images = []
   for filename in os.listdir(folder):
      img = cv2.imread(os.path.join(folder, filename),1)
      if img is not None:
        images.append(img)
   return images

def HOG_one(img,w_cell,w_block,n_bin):
    w = img.shape[0]
    h = img.shape[1]

    # matrix constrain cell (8x8)
    #đạo hàm theo x và y
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # ảnh sau đạo hàm theo trục x
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # ảnh sau đạo hàm theo trục y

    mag_3 = np.sqrt(np.square(dx) + np.square(dy))  # độ lớn bằng căn bậc hai x*x+y*y

    angel_3 = np.arctan(np.divide(dy, dx + 0.00001))  # radian
    angel_3 = np.degrees(angel_3)  # -90 -> 90
    angel_3 += 90  # 0 -> 180

    # matrix w x h x 3
    # chuan hoa dau ra đưa về ma trận  2 chiều
    # mag = max(mag_r,mag_g,mag_b)
    # angel = angel[index(max(mag_r,mag_g,mag_b))]
    # lấy giá trị lớn nhất trong r g b  theo độ lớn

    index = np.where(mag_3[:, :, 0] >= mag_3[:, :, 1], 0, 1)
    angel = np.where(mag_3[:, :, 0] >= mag_3[:, :, 1], angel_3[:, :, 0], angel_3[:, :, 1])
    mag = maximum(mag_3[:, :, 0], mag_3[:, :, 1]);

    index = np.where(mag >= mag_3[:, :, 2], index, 2)
    angel = np.where(mag >= mag_3[:, :, 2], angel, angel_3[:, :, 2])
    mag = maximum(mag, mag_3[:, :, 2])

    # tính histogram theo mag của từng cell
    # chia cell ra để láy thuộc tính cục bộ
    n_cellx = int(w / w_cell)
    n_celly = int(h / w_cell)
    hist_tensor = np.zeros([n_cellx, n_celly, n_bin])  # 16 x 8 x 9
    for cx in range(n_cellx):
        for cy in range(n_celly):
            ori_t = angel[cy * w_cell:cy * w_cell + w_cell,
                    cx * w_cell:cx * w_cell + w_cell]  # lấy ma trận 8x8 trong góc
            mag_t = mag[cy * w_cell:cy * w_cell + w_cell,
                    cx * w_cell:cx * w_cell + w_cell]  # lấy ma trận 8x8 trong độ lớn
            hist, _ = np.histogram(ori_t, bins=n_bin, range=(0, 180), weights=mag_t)  # 1-D vector, 9 elements
            #
            hist_tensor[cy, cx, :] = hist
     # print(hist_tensor.shape)
    # chuẩn hóa đầu ra
    # phép đạo hàm rất nhạy với sự thay đổi ánh sáng nên cần chuẩn hóa
    n_blockx = n_cellx - w_block + 1
    n_blocky = n_celly - w_block + 1
    deep = (w_block * w_block * n_bin)
    vector = np.zeros([n_blockx, n_blocky, deep])  # moi cell co n_bin , moi block co w_block *w_block cell

    for row in range(n_blockx):
        for col in range(n_blocky):
            arr = hist_tensor[row:row + w_block, col:col + w_block, :] # trích ra các cell
            arr = arr.reshape(deep)
            norm = ((sum(arr * arr)) ** (1 / 2))
            arr = arr / norm
            vector[row, col, :] = arr

    return vector.reshape(n_blockx*n_blocky*deep)
def HOG(img_arr,w_cell,w_block,n_bin):
    vec = []
    for i in range(len(img_arr)):
        tem = HOG_one(img_arr[i],w_cell,w_block,n_bin)
        vec.append(tem)
    return np.asarray(vec)

def distance(hist1, matrix_hist):
    dis = matrix_hist-hist1
    dis *= dis
    return sum(dis.T)

img = cv2.imread("C:\\Users\\tiem.nv164039\\Desktop\\images\\00943.png", 1);
h = HOG_one(img,8,2,9)
# w_cell =8;
# w_block = 2;
# n_bin = 9; # 0-20-...-160;
# w = img.shape[0]
# h = img.shape[1]
# d = img.shape[2]
#
# vec = HOG_one(img,w_cell,w_block,n_bin);
# print(vec.shape)
# #print(vec.shape)mg_arr = read_data("images")
# vec1 = HOG(img_arr,w_cell,w_block,n_bin);
# dis = distance(vec,vec1)
# index = np.argsort(dis)[:1]
cv2.imshow('anh goc',img)
# for i in range(len(index)):
#     cv2.imshow('anh'+str(i),img_arr[index[i]]);

H = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=((2,2)), transform_sqrt=True, block_norm="L1")
print("HOG feature: "+str(H))
print("size HOG feature: "+str(H.shape))
# print(H)
# H_arr = []
# for i in range(len(img_arr)):
#     tem = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
#                 cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
#     H_arr.append(tem)
# H_arr = np.asarray(H_arr)
# dis1 = distance(H,H_arr)
# index = np.argsort(dis1)[:2]
#
# for i in range(len(index)):
#     cv2.imshow('anh2'+str(i),img_arr[index[i]]);

cv2.waitKey(0);
cv2.destroyAllWindows();








