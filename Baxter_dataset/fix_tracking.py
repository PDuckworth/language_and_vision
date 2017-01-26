import numpy as np
import os
import shutil

def _get_f(i):
    if i < 10:
        f = "000"+str(i)
    elif i < 100:
        f = "00"+str(i)
    elif i < 1000:
        f = "0"+str(i)
    return f

folder = "138"
obj = "0"
read = 43
start = 44
finish = 229
dir = "/home/omari/Datasets/scene"+folder+"/tracking/"
file1 = _get_f(read)

srcfile = dir+"obj"+obj+"_"+file1+".pcd"
srcfile_img = dir+"obj"+obj+"_"+file1+".png"
for i in range(start,finish):
    f = open(dir+"obj"+obj+"_"+file1+".txt","r")
    file2 = _get_f(i)
    f2 = open(dir+"obj"+obj+"_"+file2+".txt","w")
    dstdir = dir+"obj"+obj+"_"+file2+".pcd"
    dstdir_img = dir+"obj"+obj+"_"+file2+".png"
    shutil.copy(srcfile, dstdir)
    shutil.copy(srcfile_img, dstdir_img)
    for line in f:
        f2.write(line)
    f2.close()
    f.close()
