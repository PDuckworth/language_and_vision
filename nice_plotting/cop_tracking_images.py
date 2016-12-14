import cv2
import numpy as np

dir1 = "/home/omari/Baxter_Dataset_Final/scene226/tracking/scene_"
dir2 = "/home/omari/Baxter_Dataset_Final/scene226/kinect_rgb/Kinect_"
dir3 = "/home/omari/Baxter_Dataset_Final/scene226/tracking/combined_"

for i in range (1,190):
    if i < 10:
        file = '000'+str(i)
    elif i < 100:
        file = '00'+str(i)
    elif i < 1000:
        file = '0'+str(i)
    img = cv2.imread(dir1+file+'.png')
    img = img[:,320:1180,:]
    img2 = cv2.imread(dir2+file+'.png')
    print img.shape
    img2 = cv2.resize(img2,None,fx=577.0/540, fy=577.0/540, interpolation = cv2.INTER_CUBIC)
    print img2.shape
    vis = np.concatenate((img2[:,30:840,:], img), axis=1)
    cv2.imshow('img',vis)
    cv2.imwrite(dir3+file+'.png',vis)
    cv2.waitKey(10)
