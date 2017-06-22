import cv2
import numpy as np
import glob
from operator import add
import pickle
from sklearn import mixture
# import itertools
from sklearn.metrics.cluster import v_measure_score
from sklearn import metrics
from sklearn import svm

class language():
    """docstring for shapes."""
    def __init__(self):
        self.dir = "/home/omari/Datasets/Baxter_Dataset/scene"
        self.dir_save = "/home/omari/Datasets/Baxter_Dataset/features/language/"
        self.unique_words = []

    def _read_sentences(self):
        for video in range(1,205):
            dir1 = self.dir+str(video)+"/annotation/natural_language_commands.txt"
            f = open(dir1,'r')
            for line in f:
                line = line.split("\n")[0]
                print video,line
                words = line.split(" ")
                for word in words:
                    if word not in self.unique_words:
                        self.unique_words.append(word)
        print len(self.unique_words)


def main():
    L = language()
    L._read_sentences()

if __name__=="__main__":
    main()
