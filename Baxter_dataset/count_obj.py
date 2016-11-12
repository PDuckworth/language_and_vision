import numpy as np
import glob, os

dir1 = '/media/omari/Elements/Baxter_Dataset_Final/scene'

count = 0
for scene in range(1,205):
    d = dir1+str(scene)+'/clusters/'
    os.chdir(d)
    for file in glob.glob("cloud*"):
        count+=1

print count/204.0
