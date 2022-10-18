import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
đọc ảnh gốc , đọc ảnh mẫu 
tìm histogram ảnh gốc , histogram ảnh mẫu 
với mỗi điểm mức sáng của ảnh gốc  tìm histogram gần nhất trong ảnh mẫu 
chuyển mức sáng tương ứng về mức sáng trong ảnh mẫu.
 
'''
#plt.hist(img)
img = cv2.imread("C:\\Users\\tiem.nv164039\\PycharmProjects\\XuLiAnh\\img\\bean.jpg", 0);
print(img)
cv2.imshow('linh',img);
#hist = cv2.calcHist(img,[0],None,[256],[0,256])



cv2.waitKey(0);
cv2.destroyAllWindows();
