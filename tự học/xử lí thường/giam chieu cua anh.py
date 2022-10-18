import cv2
import numpy as np
from numpy import maximum


def convert(img):
    # define a threshold, 128 is the middle of black and white in grey scale
    thresh = 120
    # threshold the image
    img_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    # save image
    return img_binary;

def incre(img,w,p):
    r,c,d = img.shape
    b = np.zeros((r+2,c+2,d))
    b[1:r+1,1:c+1,:] =img
    img = b
    r, c,d = b.shape
    r= int((r-w)/p+1)
    c = int((c-w)/p+1)
    m=0;
    n=0
    imgn = np.zeros((r,c,d))
    for v in range(3):
     for i in range(r):
        for j in range(c):
         ker= img[i*p:i*p+w,j*p:j*p+w,v]
         ker = ker.reshape(-1)
         imgn[i][j][v] =(sum(ker)/(w*w))                              #sum(ker)/9;
    return imgn

def convert_1(img):
  img = np.asarray(img)

  img = np.where(img==255,1,0) # convert ve 0 va 1

  r, c,d = img.shape

  for y in range(0, r):
    for x in range(0, (c + 7)//8 * 8):
      if x == 0:
        print("  ", end='')
      if x % 8 == 0:
        print("0b", end='')

      bit = '0'

      if x < c and img[y][x] != 0:
        bit = '1'
      print(bit, end='')

      if x % 8 == 7:
        print(",", end='')
    print()


imgn= cv2.imread("11.jpg", 1);
imgn = incre(imgn,2,2)

# imgn = incre(imgn,2,2)
# imgn = incre(imgn,2,2)

# imgn = incre(imgn,2,2)
# imgn = incre(imgn,3,1)
# imgn = incre(imgn,3,1)
# imgn = incre(imgn)
# imgn = imgn.T
# imgn = img2=cv2.flip(imgn,0)
imgn = np.uint8(imgn)
# imgn =convert(imgn)
#r ,c  = imgn.shape
result = imgn  #[56:118,:]           #[51:115,:]                  #[int(r/2)-5:r-5,:]
#convert_1(result)
print(result.shape)
# img = np.where(imgn==255,0,1) # convert ve 0 va 1
# print(img)
# np.savetxt("img.txt", img,fmt='%1u')
cv2.imwrite('anh9.jpg', result)
# # cv2.imshow("or",img)
cv2.imshow("bi",result)
cv2.waitKey(0);
cv2.destroyAllWindows();