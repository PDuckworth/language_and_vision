import numpy as np
import itertools

from scipy import linalg
import matplotlib as mpl
#
import getpass
import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import colorsys
from sklearn import mixture

import pickle

class clustering():
    """docstring for clustering"""
    def __init__(self):
        self.username = getpass.getuser()
        self.dir1 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/features/vid'
        self.dir2 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/clusters/colours/'
        self.folder = 1

    def _read_colours(self):
        f = open(self.dir2+'colour_clusters.txt','w')
        self.X,self.Y,self.Z,self.c,self.R,self.G,self.B,clf,Y_ = pickle.load(open( self.dir2+"colours.p", "rb" ) )
        colours = {}
        for i in range(1,494):
            # print (i-1)*2,i*2
            c = Y_[(i-1)*2:i*2]
            colours[i] = c
            f.write('vid_'+str(i)+':'+str(c[0])+','+str(c[1])+'\n' )
        f.close()
        pickle.dump( colours, open( self.dir2+"colours_clusters.p", "wb" ) )



def main():
    f = clustering()
    f._read_colours()
    # f.create_sk_images()

if __name__=="__main__":
    main()
