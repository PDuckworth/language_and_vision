import numpy as np
import os
import shutil
import glob
import cv2
# import zipfile

counter = 0
frames = 0
for k in range(1,6):
    dir_ = "/home/omari/Datasets/jivko_dataset/t"+str(k)+"/obj_"
    for i in range(1,33):
        folder = str(i)
        dir_dst = dir_+folder+"/trial_1/"

        for d in ["drop","grasp","hold","lift","lower","press","push"]:
            files = glob.glob(dir_dst+d+"/vision_data/*.jpg")
            frames += len(files)
            print frames
