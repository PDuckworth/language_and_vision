import numpy as np
import itertools

import matplotlib as mpl
#
import getpass
import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import colorsys
import sys
import pickle

class association():
    """docstring for clustering"""
    def __init__(self):
        self.username = getpass.getuser()
        self.dir_colour = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/clusters/colours/'
        self.dir_text = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/ECAI_annotations/vid'
        self.folder = 1

        self.good_videos = "1 6 13 14 18 19 20 21 26 27 28 29 163 184 186 202 322 340 5 9 12 22 200 364 481 489 8 15 30 32 37 49 92 109 111 117 217 298 358 389 460 464 469 472 477 484 50 145 309 360 375 445 454 459 483 52 81 115 253 271 284 297 308 418 435 59 66 125 277 302 386 89 90 274 307 363 379 316 324 357 366 455"
        self.good_videos = self.good_videos.split(' ')
        self.good_videos = map(int, self.good_videos)

    def _read_colours(self):
        # self.colours = pickle.load(open( self.dir_colour+"colours_clusters.p", "rb" ) )
        self.colours = {}
        f = open(self.dir_colour+'colour_clusters_GT.txt','r')
        for line in f:
            line = line.split('\n')[0].split(':')
            vid = int(line[0].split('_')[1])
            colours = line[1].split(',')
            self.colours[vid] = colours

        for key in self.colours:
            # get only the unique colours
            # if self.colours[key][0] == 5: self.colours[key][0] = 0
            # if self.colours[key][1] == 5: self.colours[key][1] = 0
            # if self.colours[key][0] == 6: self.colours[key][0] = 1
            # if self.colours[key][1] == 6: self.colours[key][1] = 1

            if self.colours[key][0] == self.colours[key][1]:
                self.colours[key] = [self.colours[key][0]]

    def _read_annotations(self):
        self.words = {}
        for i in range(1,494):
            self.words[i] = []
            f = open(self.dir_text+str(i)+'/person.txt','r')
            for count,line in enumerate(f):
                if count == 0 or "(X)" in line or line == "\n":
                    continue
                line = line.split('\n')[0]
                if '#' in line:
                    print line
                    sys.exit(1)
                line = line.lower()
                line.replace('.','',20)
                line.replace(',','',20)
                for word in line.split(' '):
                    if word != '' and word not in self.words[i]:
                        self.words[i].append(word)
            # print self.words[i]
            # print i, '------------'

    def _build_associations(self):
        self.association = {}
        # self.association['colour'] = {}
        for i in range(1,494):
            # if i not in self.good_videos: continue
            # if 'blue' in self.words[i]:
            #     if 'blue' not in self.colours[i]:
                    # print '>>',i
            for word in self.words[i]:
                if word not in self.association:
                    self.association[word] = {}
                    self.association[word]['N'] = 0.0
                    self.association[word]['colour'] = {}

                self.association[word]['N'] += 1.0
                for colour in self.colours[i]:
                    if colour not in self.association[word]['colour']:
                        self.association[word]['colour'][colour] = 0
                    self.association[word]['colour'][colour] += 1


        for word in self.association:
            # if word != 'blue':
            #     continue
            if self.association[word]['N']>10:
                print '## ',word
                A = sorted(self.association[word]['colour'], key=lambda k: self.association[word]['colour'][k])
                for colour in  reversed(A):
                    value = self.association[word]['colour'][colour]/self.association[word]['N']
                    if value>.7:
                        print colour,':',value
                print '----------'



def main():
    f = association()
    f._read_colours()
    f._read_annotations()
    f._build_associations()
    # f.create_sk_images()

if __name__=="__main__":
    main()
