import numpy as np
import os
import shutil
import glob
import cv2

for i in range(166,205):
    folder = str(i)
    dir_src = "/home/omari/Datasets/Baxter_Dataset_final/scene"+folder
    dir_dst = "/home/omari/Datasets/AAAI-17-Baxter-dataset/scene"+folder
    if not os.path.exists(dir_dst):
        os.makedirs(dir_dst)

    # if not os.path.exists(dir_dst+"/cam"):
    #     os.makedirs(dir_dst+"/cam")
    for s,d in zip(["/cam","/kinect_rgb","/clusters","/LH_rgb","/RH_rgb","/Robot_state","/table_pc","/tabletop_pc","/tracking"],["/rgb_cam","/rgb_kinect","/objects","/rgb_lefthand","/rgb_righthand","/robot_state","/table_pointcloud","/tabletop_pointcloud","/object_tracks"]):
        try:
            shutil.copytree(dir_src+s,dir_dst+d)
        except Exception as e:
            print 'file exists'


    if not os.path.exists(dir_dst+"/annotation"):
        os.makedirs(dir_dst+"/annotation")
    src = "/home/omari/Dropbox/Baxter_dataset/data/"+folder+"_commands_test.txt"
    dst = dir_dst+"/annotation/natural_language_commands.txt"
    shutil.copyfile(src, dst)
