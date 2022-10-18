import numpy as np
import cv2

'''
thuat toan bayes
su dung HOG 
hogFaceDetector = dlib.get_frontal_face_detector()
faceRects = hogFaceDetector(frameDlibHogSmall, 0)
for faceRect in faceRects:
    x1 = faceRect.left()
    y1 = faceRect.top()
    x2 = faceRect.right()
    y2 = faceRect.bottom()

'''



# Bước 1: Tấm ảnh và tệp tin xml
face_cascade = cv2.CascadeClassifier("Haar Cascade/haarcascade_frontalface_default.xml")
image = cv2.imread("t.png")
print(image)
cv2.imshow("Faces found", image)
# Bước 2: Tạo một bức ảnh xám
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bước 3: Tìm khuôn mặt
faces = face_cascade.detectMultiScale(
    grayImage,
    scaleFactor=1.1,
    minNeighbors=3
)
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# Bước 4: Vẽ các khuôn mặt đã nhận diện được lên tấm ảnh gốc
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# for (x,y,w,h) in faces:
#     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = grayImage[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Bước 5: Vẽ lên màn hình
cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
