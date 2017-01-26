import numpy as np
import cv2

dir1 = '/home/omari/Dropbox/Reports/AAAI17/camera_ready/pics/not_used/'
# print img.shape
sp = 20
total_img = np.zeros((2800,600*4+sp*3,3),dtype=np.uint8)+255
c = 0
for i in ['021','045','130','182']:
    img = cv2.imread(dir1+'Kinect_0'+i+'.png')
    print img.shape
    img = img[120:,110:720,:]
    img = cv2.resize(img,None,fx=600.0/img.shape[1], fy=600.0/img.shape[1], interpolation = cv2.INTER_CUBIC)
    # print img.shape
    total_img[0:img.shape[0],600*c+sp*c:600*(c+1)+sp*c,:] = img
    c+=1


y = img.shape[0]
total_img[y+8:y+12,10:-10,:] = 0
y+=10
c = 0
for i in ['0','1','2','3']:
    img = cv2.imread(dir1+'cloud_cluster_'+i+'.png')
    print img.shape
    img = img[120:580,:,:]
    img = cv2.resize(img,None,fx=600.0/img.shape[1], fy=600.0/img.shape[1], interpolation = cv2.INTER_CUBIC)
    # print img.shape
    total_img[y+10:y+10+img.shape[0],600*c+sp*c:600*(c+1)+sp*c,:] = img
    c+=1

y += img.shape[0]
total_img[y+8:y+12,10:-10,:] = 0
y+=10
c = 0
for i in ['021','045','130','182']:
    img = cv2.imread(dir1+'scene_0'+i+'.png')
    # print img.shape
    img = img[100:,390:1100,:]
    img = cv2.resize(img,None,fx=600.0/img.shape[1], fy=600.0/img.shape[1], interpolation = cv2.INTER_CUBIC)
    # print img.shape
    total_img[y+10:y+10+img.shape[0],600*c+sp*c:600*(c+1)+sp*c,:] = img
    c+=1


y += img.shape[0]
total_img[y+8:y+12,10:-10,:] = 0
y+=10
c = 0
for i in ['0','1','2','3']:
    img = cv2.imread(dir1+'HSV_cluster_'+i+'.png')
    # print dir1+'HSV_cluster_ '+i+'.png'
    # print img
    # print img.shape
    img = img[100:500,:,:]
    img = cv2.resize(img,None,fx=600.0/img.shape[1], fy=600.0/img.shape[1], interpolation = cv2.INTER_CUBIC)
    # print img.shape
    total_img[y+10:y+10+img.shape[0],600*c+sp*c:600*(c+1)+sp*c,:] = img
    c+=1



total_img = total_img[:,:,:]
cv2.imwrite(dir1+'example_long.png',total_img)
total_img = cv2.resize(total_img,None,fx=1000.0/total_img.shape[1], fy=1000.0/total_img.shape[1], interpolation = cv2.INTER_CUBIC)
cv2.imshow('img',total_img)
cv2.waitKey(2000)
