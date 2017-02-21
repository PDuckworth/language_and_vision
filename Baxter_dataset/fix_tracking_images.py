import numpy as np
import os
import shutil
import glob
import cv2

def _get_f(i):
    if i < 10:
        f = "000"+str(i)
    elif i < 100:
        f = "00"+str(i)
    elif i < 1000:
        f = "0"+str(i)
    return f

for i in range(1,205):
    folder = str(i)
    dir = "/home/omari/Datasets/Baxter_Dataset_final/scene"+folder+"/tracking/"
    images = sorted(glob.glob(dir+"*.png"))
    for img in images:
        im = cv2.imread(img)
        # black = np.dot(im[:,:,0]==0,im[:,:,1]==0)
        # print black
        # black = np.dot(black,im[:,:,2]==0)
        # im[black] = 255
        # rows,cols,_ = im.shape
        # M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        # im = cv2.warpAffine(im,M,(cols,rows))
        # cv2.imwrite(img,im)

        # cv2.imshow('img',im)
        # cv2.waitKey(10)
