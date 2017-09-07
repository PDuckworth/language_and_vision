import numpy as np
from sklearn import mixture
import itertools
from sklearn.metrics.cluster import v_measure_score
import cv2
import pickle
#import pulp
import sys
import colorsys
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import shuffle

class objects_class():
    """docstring for faces"""
    def __init__(self):
        # self.username = getpass.getuser()
        # self.dir1 = '/home/'+self.username+'/Datasets/ECAI_dataset/features/vid'
        self.dir2 = '/home/omari/Datasets/ECAI Data/annotation/vid'
        self.dir_objects =  '/home/omari/Datasets/ECAI_dataset/features/vid'
        self.dir_grammar = '/home/omari/Datasets/ECAI_dataset/grammar/'
        self.dir_annotation = '/home/omari/Datasets/ECAI_dataset/ECAI_annotations/vid'
        self.im_len = 60
        self.f_score = []
        self.Pr = []
        self.Re = []
        self.ok_clusters = []
        self.ok_videos = []


    def _read_objects(self):
        counter = 0
        for i in range(1,494):
            f = open(self.dir2+str(i)+"/objects.txt","r")
            for line in f:
                counter += 1
                print line
        print counter/493.0

def main():
    f = objects_class()
    f._read_objects()

if __name__=="__main__":
    main()
