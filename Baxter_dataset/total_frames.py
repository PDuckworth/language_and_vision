import numpy as np
import cv2
import os, os.path
# dir_1 = '/home/omari/Dropbox/Posters/1st_UK_workshop/pics/'
#
# images = ['Kinect_0017','Kinect_0025','Kinect_0036','Kinect_0055']

dir_1 = '/media/omari/Elements/Baxter_Dataset_Final/scene'

count = 0
for i in range(1,205):
    DIR = dir_1+str(i)+'/cam/'
    print DIR
    # path joining version for other paths
    # DIR = '/tmp'
    count += len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
print count
