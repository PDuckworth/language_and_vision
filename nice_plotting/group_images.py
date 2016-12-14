import numpy as np
import cv2

dir1 = '/home/omari/Dropbox/Reports/AAAI17/camera_ready/pics/'
# print img.shape
sp = 20
total_img = np.zeros((806,480*4+sp*3,3),dtype=np.uint8)+255
c = 0
for i in [1,2,4,5]:
    img = cv2.imread(dir1+'IMG_'+str(i)+'.jpg')
    img = cv2.resize(img,None,fx=.2, fy=.2, interpolation = cv2.INTER_CUBIC)
    img = img[:,75:555,:]
    total_img[:,480*c+sp*c:480*(c+1)+sp*c,:] = img
    c+=1


total_img = total_img[50:,:,:]
cv2.imshow('img',total_img)
cv2.imwrite(dir1+'example4.png',total_img)
cv2.waitKey(2000)
