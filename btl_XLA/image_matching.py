from numpy import linalg as la
import numpy as np
import math
import cv2
import os
import math
from numpy import maximum

a = np.zeros(9);
a[0] = 2**0;
a[1] = 2**1
a[2] = 2**2;
a[3] = 2**7;
a[4] = 0;
a[5] =2**3;
a[6] = 2**6;
a[7] = 2**5;
a[8] = 2**4;

index_arr = np.array(
    [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127,
     128, 129,131,135,143,159,191,192,193,195,199,207,223,224,225,227,231,239,240,241,243,247,248,249,251,252,253,254,255]);


# read image  , each image a row
def read_data_color(folder):
   images = []
   N = []
   for filename in os.listdir(folder):
      img = cv2.imread(os.path.join(folder, filename),1)

      if img is not None:

        images.append(img)
   return images;

def getHis_one(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    color = ('H', 'S', 'V')
    his = [];

    n = (img.shape[0]*img.shape[1])
    for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])  # chanel 0 - 1 -2 cac kenh cua anh
        histr /=n;
        his.append(histr);
    his = np.asarray(his).reshape(3,-1)
    return his

def getHist(matrix):
    his =[]
    for i in range(len(matrix)):
        hist = getHis_one(matrix[i])
        his.append(hist)
    return np.asarray(his)

# convolution image
def cov(a,matrix):
    n = matrix.shape[0];
    h = a.shape[0];
    w = a.shape[1]
    b = np.zeros((h+2*(n-2),w+2*(n-2)))
    b[n-2:h+n-2,n-2:w+n-2] =a;
    re = np.zeros((h,w));
    for i in range(n-2,h+n-2):
        for j in range(n - 2, w + n - 2):
            print((matrix*b[i-(n-2):i+(n-2)+1,j-(n-2):j+(n-2)+1]))
            re[i-(n-2)][j-(n-2)] = sum(sum(matrix*b[i-(n-2):i+(n-2)+1,j-(n-2):j+(n-2)+1]))
    print(re)
    return re;

# moment
def getVector_one(hist):
#kich thuoc 3*256
    result = []
    index =[]
    index_test = np.arange(256)
    index.append(index_test)
    index.append(index_test)
    index.append(index_test)
    tem =hist*index  # xác suât nhân với giá trị
    E=sum(tem.T).reshape(3,1)     # tính kì vọng
    E/=255
    index = np.asarray(index)
    sub = index/255-E                # i =0 -> 255 - trung binh
    E = E.reshape(-1)
    do_lech = (((sum(((sub**2)*hist).T)))**(1/2)) # do lech chuan
    ttt= sum(((sub**3)*hist).T)
    tem = np.where(ttt<0,-1,1)
    Skewness = ((ttt*tem)**(1/3))*tem
    result.append(E)
    result.append(do_lech)
    result.append(Skewness)
    return (np.asarray(result).T).reshape(-1)

def getVetor(hist):
    n = len(hist);
    vec=[]
    for i in range(n):
        his = getVector_one(hist[i])
        vec.append(his)
    return np.asarray(vec)

def HOG_one(img,w_cell,w_block,n_bin):
    w = img.shape[0]
    h = img.shape[1]
    d = img.shape[2]
    # matrix constrain cell (8x8)
    # chuan hoa dau vao
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # ảnh sau đạo hàm theo trục x
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # ảnh sau đạo hàm theo trục y
    mag_3 = np.sqrt(np.square(dx) + np.square(dy))  # độ lớn bằng căn bậc hai x*x+y*y
    angel_3 = np.arctan(np.divide(dy, dx + 0.00001))  # radian
    angel_3 = np.degrees(angel_3)  # -90 -> 90
    angel_3 += 90  # 0 -> 180
    # matrix w x h x 3
    # chuan hoa dau ra đưa về ma trận một 2 chiều
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
    hist_tensor = np.zeros([n_cellx, n_celly, n_bin])  #
    for cx in range(n_cellx):
        for cy in range(n_celly):
            ori_t = angel[cy * w_cell:cy * w_cell + w_cell,
                    cx * w_cell:cx * w_cell + w_cell]  # lấy ma trận 8x8 trong góc
            mag_t = mag[cy * w_cell:cy * w_cell + w_cell,
                    cx * w_cell:cx * w_cell + w_cell]  # lấy ma trận 8x8 trong độ lớn
            hist, _ = np.histogram(ori_t, bins=n_bin, range=(0, 180), weights=mag_t)  # 1-D vector, 9 elements
            #
            hist_tensor[cy, cx, :] = hist

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

def HOG(img_maxtrix,w_cell,w_block,n_bin):
    vec = [] ;
    for i in range(len(img_maxtrix)):
        tem = HOG_one(img_maxtrix[i],w_cell,w_block,n_bin)
        vec.append(tem)
    return np.asarray(vec)

def read_data_gray(folder):
   images = []
   for filename in os.listdir(folder):
      img = cv2.imread(os.path.join(folder, filename),0)
      if img is not None:
        images.append(img)
   return images

# LPB with one image
def lbp_one(img,a,index_arr):
    r, v = img.shape; # get  shape
    new_img = np.zeros((r + 2, v + 2)); # generate array
    new_img[1:r + 1, 1:v + 1] = img; #
    his_lbp_one = np.zeros(256);
    max = 0;
    for i in range(1, r + 1):
        for j in range(1, v + 1):
            mask = new_img[i - 1:i + 2, j - 1:j + 2] - new_img[i][j];
            mask = np.where(mask >= 0, 0, 1)
            tem = mask.reshape(9) * a;
            index = int(sum(tem))
            his_lbp_one[index] += 1

    t = sum(his_lbp_one)

    his_lbp = np.zeros(59);
    his_lbp[0:58] = his_lbp_one[index_arr]
    his_lbp[58] = t - sum(his_lbp_one[index_arr])
    return (his_lbp)/(r*v) #his_lbp       #

def lbp(img_arr, a, index_arr):
    his = [] ;
    for i in range(len(img_arr)):
        tem  = lbp_one(img_arr[i],a,index_arr);
        his.append(tem)
    return np.asarray(his);

def distance_eu(hist1, matrix_hist):
    dis = matrix_hist-hist1
    dis *= dis
    dis = sum(dis.T)
    dis = dis **(1/2)
    return dis/hist1.shape[0]

def distance_cos(hist1,matrix_hist):
    dis = matrix_hist*hist1
    size_1 = (sum(hist1*hist1))**(1/2)# 1
    size_v = matrix_hist*matrix_hist
    size_v = sum(size_v.T)
    size_v = size_v**(1/2)
    dis =sum(dis.T)
    dis = dis/(size_v*size_1);
    return dis


def begin(img_path,arr_img_path,feature,distance_type):
    #read data
    path =[]
    arr_img =[]
    arr_img_gray=[]
    img = cv2.imread(img_path,1)
    img_gray = cv2.imread(img_path,0)
    for i in range(len(arr_img_path)):
        tem = cv2.imread(arr_img_path[i],1)
        tem1 = cv2.imread(arr_img_path[i], 0)
        arr_img.append(tem)
        arr_img_gray.append(tem1)
    arr_img =np.asarray(arr_img)
    arr_img_gray = np.asarray(arr_img_gray)
    #print(arr_img.shape)
    n = len(feature)
    #print(feature)

    fea_one = np.zeros((1), dtype=int)

    # fea = np.asarray(fea)
    fea = np.zeros((len(arr_img_path),1),dtype=int)
    for i in range(n):
       if feature[i]=="CH":
           print("CH")
           his_vector_one = getHis_one(img)
           his_vector = getHist(arr_img);
           his_vector_one= his_vector_one.reshape(-1)
           his_vector=his_vector.reshape((len(arr_img_path),-1))
           fea_one=np.append(fea_one, his_vector_one)
           # print(fea_one.shape)
           fea =np.append(fea, his_vector, axis=1)
           # print(fea.shape)
       if feature[i]=="CM":
           print("CM")
           his_vector_one = getHis_one(img)
           his_vector = getHist(arr_img);
           mmt_vector_one = getVector_one(his_vector_one)
           mmt_vector = getVetor(his_vector)
           fea_one=np.append(fea_one, mmt_vector_one)
           # print(fea_one.shape)
           fea = np.append(fea, mmt_vector, axis=1)
           # print(fea.shape)
       if feature[i]=="LBP":
            print("LBP")
            lbp_vector_one = lbp_one(img_gray, a, index_arr)
            lbp_vector = lbp(arr_img_gray, a, index_arr)
            fea_one=np.append(fea_one, lbp_vector_one)
            # print(fea_one.shape)
            fea=np.append(fea, lbp_vector, axis=1)
            # print(fea.shape)
       if feature[i]=="HOG":
           print("HOG")
           hog_vector_one = HOG_one(img, 8, 2, 9)
           hog_vector = HOG(arr_img, 8, 2, 9);
           fea_one =  np.append(fea_one, hog_vector_one)
           # print(fea_one.shape)
           fea=np.append(fea, hog_vector, axis=1)
           # print(fea.shape)
    w = fea_one.shape[0]
    r,v = fea.shape
    re_one = np.zeros(w-1)
    re = np.zeros((r,v-1))
    re_one = fea_one[1:]
    re = fea[:,1:]
    print(re_one.shape)
    print(re.shape)
    if distance_type=="EU":
        dis =distance_eu(re_one,re)
        # print(dis)
        index = np.argsort(dis)[:1]
        # print(index)
        for i in range(len(index)):
            cv2.imshow('anh' + str(i), arr_img[index[i]]);
            print(dis[index[i]])

    elif distance_type=="CO":
         dis = distance_cos(re_one, re)
         index = np.argsort(dis)[-1:]
         index = index[::-1]
         # print(dis[i])
         for i in range(len(index)):
             cv2.imshow('anh' + str(i), arr_img[index[i]]);
             print(dis[index[i]])
    cv2.imshow('anh goc', img)
    cv2.waitKey(0);
    cv2.destroyAllWindows();

path_one = "C:\\Users\\tiem.nv164039\\Desktop\\images\\xam.png"
img = cv2.imread(path_one,1);
path_arr =[]
# pa
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00960.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00944.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00945.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00946.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00947.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00948.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00949.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00950.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00951.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00952.png")
path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\trang.png")
# path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00954.png")
# path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00955.png")
# path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00956.png")
# path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00957.png")
# path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00958.png")
# path_arr.append("C:\\Users\\tiem.nv164039\\Desktop\\images\\00959.png")

feature =[]
# feature.append("CH")
feature.append("CM")
# feature.append("LBP")
# feature.append("HOG")

distance_type = "EU"
distance_type1 = "CO"
begin(path_one,path_arr,feature,distance_type1)


