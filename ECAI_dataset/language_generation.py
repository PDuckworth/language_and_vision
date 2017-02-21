import numpy as np
from sklearn import mixture
import itertools
from sklearn.metrics.cluster import v_measure_score
from sklearn import metrics
import cv2
import pickle
import pulp
import sys
import copy
import colorsys

import numpy as np
import matplotlib.pyplot as plt

class language_generation_class():
    """docstring for faces"""
    def __init__(self):
        self.dir_faces =  '/home/omari/Datasets/ECAI_dataset/faces/'
        self.dir_colours =  '/home/omari/Datasets/ECAI_dataset/colours/'
        self.dir_colours2 =  '/home/omari/Datasets/ECAI_dataset/features/vid'
        self.dir_grammar = '/home/omari/Datasets/ECAI_dataset/grammar/'
        self.dir_annotation = '/home/omari/Datasets/ECAI_dataset/ECAI_annotations/vid'
        self.im_len = 60
        self.videos_to_process = [35,481,441]
        self.faces_X = {}
        self.faces_results = {}
        self.colours_X = {}
        self.colours_results = {}

    def _read_faces_clusters(self):
        self.faces,self.faces_clf,self.X,self.best_v = pickle.load(open(self.dir_faces+'faces_clusters.p',"rb"))
        self.faces_assignments = pickle.load(open(self.dir_faces+'faces_assignments.p'))

    def _read_faces(self):
        f = open(self.dir_faces+'faces3_projections.csv','rb')
        f2 = open(self.dir_faces+'faces_images.csv','rb')
        for line1,line2 in zip(f,f2):
            line1 = line1.split('\n')[0]
            line2 = line2.split('\n')[0].split(".")[0].split("/")[1].split("_")[1]
            vid = int(line2)
            if vid in self.videos_to_process:
                if vid not in self.faces_X:
                    self.faces_X[vid] = []
                data = line1.split(',')[1:]
                data = map(float, data)
                if self.faces_X[vid] == []:
                    self.faces_X[vid].append(data)
                else:
                    self.faces_X[vid] = np.vstack((self.faces_X[vid],data))

    def _get_name_of_person(self):
        for video in self.faces_X:
            if video not in self.faces_results:
                self.faces_results[video] = []
            for cluster in self.faces_clf.predict(self.faces_X[video]):
                self.faces_results[video].append(self.faces_assignments[cluster][0])
        for video in self.faces_results:
            print video,self.faces_results[video]
            print '----------'

    def _read_colours_clusters(self):
        self.colours_clf,X_ = pickle.load(open(self.dir_colours+'colour_clusters.p',"rb"))
        self.colours_assignments = pickle.load(open(self.dir_colours+'colours_assignments.p'))

    def _read_colours(self):
        for f1 in self.videos_to_process:
            if f1 not in self.colours_X:
                self.colours_X[f1] = []
            count1 = 0
            count2 = 0
            top = 1
            f = open(self.dir_colours2+str(f1)+'/colours.txt','r')
            for line in f:
                line = line.split('\n')[0]
                if '-' in line:
                    top = 0
                else:
                    data = []
                    line = line.split(',')
                    if top and count1<1:
                        count1+=1
                        data = map(int, line)

                    if not top and count2<1:
                        count2+=1
                        data = map(int, line)

                    if data != []:
                        hls = colorsys.rgb_to_hls(data[2]/255.0, data[1]/255.0, data[0]/255.0)
                        xyz = self._hls_to_xyz(hls)
                        if self.colours_X[f1] == []:
                            self.colours_X[f1] = [xyz]
                        else:
                            self.colours_X[f1] = np.vstack((self.colours_X[f1],xyz))

    def _get_colour_of_person(self):
        for video in self.colours_X:
            if video not in self.colours_results:
                self.colours_results[video] = []
            for cluster in self.colours_clf.predict(self.colours_X[video]):
                print video,cluster
                self.colours_results[video].append(self.colours_assignments[cluster])
        for video in self.colours_results:
            print self.colours_results[video]
            print '----------'

    def _hls_to_xyz(self,hls):
        h = hls[0]*2*np.pi
        l = hls[1]
        s = hls[2]
        z = l
        x = s*np.cos(h)
        y = s*np.sin(h)
        return [x,y,z]

def main():
    f = language_generation_class()
    f._read_faces_clusters()
    f._read_faces()
    f._get_name_of_person()
    f._read_colours_clusters()
    f._read_colours()
    f._get_colour_of_person()
    


if __name__=="__main__":
    main()
