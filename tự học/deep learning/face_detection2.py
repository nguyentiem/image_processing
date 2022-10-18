import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# def read_data(folder):
# images = []
face_cascade = cv2.CascadeClassifier('Haar Cascade/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('Haar Cascade/haarcascade_eye.xml')
i=0
for filename in os.listdir("../images/"):
  img = cv2.imread(os.path.join("../images/", filename),1)
  if img is not None:
     i+=1
    # images.append(img)
    # return images



# img = cv2.imread('../images/mt.jpg')

     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

     faces = face_cascade.detectMultiScale(
             gray,
             scaleFactor=1.1,
             minNeighbors=1
     )
     for (x,y,w,h) in faces:
        # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        # roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        cv2.imwrite("img"+str(i)+".jpg",roi_color)
        cv2.imshow("img" + str(i) , roi_color)
                # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.waitKey(0)
cv2.destroyAllWindows()
