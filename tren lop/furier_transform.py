import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
pho pha la ket cau anh 
pho bien do la do sang toi 
lay nguong tan so 
nhan anh bien foi voi dap ung tan so 
convert nguoc   
/// loc  
'''
img  = np.zeros((100,100));

img[25:75,25:75]=1 ;

fft  = np.fft.fft2(img)
fft = np.fft.fftshift(fft)
re = np.real(fft)
im = np.imag(fft)
result = 20 * np.log(cv2.magnitude(re, im))

ffta  = np.uint8(abs(fft))

# angle(A) : its phase spectrum
cv2.imshow("or",img);

cv2.imshow("r",ffta);

cv2.imshow("r1",np.uint8(result));
#plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();
