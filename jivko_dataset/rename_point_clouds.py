import os
import glob

for t in range(1,5):
    for o in range(1,33):
        dir1 = "/home/omari/Datasets/jivko_dataset/t"+str(t)+"/obj_"+str(o)+"/trial_1/look/vision_data/test_"
        unique_objects = sorted(glob.glob(dir1+"*.pcd"))
        if len(unique_objects)==1:
            print unique_objects[0]
            os.rename(unique_objects[0], dir1+".pcd")
        else:
            print "WTFFFF",t,o
