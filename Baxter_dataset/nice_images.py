import numpy as np
import cv2

# dir_1 = '/home/omari/Dropbox/Posters/1st_UK_workshop/pics/'
#
# images = ['Kinect_0017','Kinect_0025','Kinect_0036','Kinect_0055']

dir_1 = '/home/omari/Dropbox/Reports/AAAI17/grounding_language_to_vision/pics_not_used/'

images = ['cam_0020','cam_0031','cam_0047']


for img in images:
    im = cv2.imread(dir_1+img+'.png')
    im = im[:600,20:420,:]
    cv2.imshow('test',im)
    cv2.waitKey(1000)
    cv2.imwrite(dir_1+img+'_m.png',im)
