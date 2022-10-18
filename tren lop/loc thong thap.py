import cv2
import numpy as np
import matplotlib.pyplot as plt
D0 = 100;
img  = cv2.imread("1.png",0);
fft  = np.fft.fft2(img); # furier transform
bien_do = 20 * np.log(cv2.magnitude(fft[:, :, 0], fft[:, :, 1]))
# fshift   = np.fft.fftshift(img)
# tao mang chi so
r,c = img.shape
a = np.arange(r).reshape(r,1);
b = np.arange(c).reshape(1,c)
M = r/2
N = c/2
D = ((N)**(2)+(M)**(2))**(1/2)
D = D*D0/100

a =(a-M)**(2);
b=(b-N)**2;
d = (a+b)**(1/2)
H = np.where(d<D,1,0)
fft1 = H*fft
imgn = np.fft.ifft2(fft1)
imgn  = np.uint8(imgn)
# #
# ffta = np.uint8(fft)

# tinh H : dap ung tan so
# tap ra gia tri u v
# tinh D khoang cach tu uv den tam
#


cv2.imshow("or",img);
cv2.imshow("r",imgn);
#plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();