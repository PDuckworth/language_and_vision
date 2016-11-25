import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

dir1 = "/home/omari/Datasets_old/Jivko_dataset/t1/obj_"
dir2 = "/trial_1/drop/vision_data"

for obj in range(33):
    mypath = dir1+str(obj)+dir2
    imgs = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
    img = cv2.imread(mypath+'/'+imgs[0])
    cv2.imwrite("/home/omari/Datasets_old/Jivko_dataset/objects/"+imgs[0],img)
