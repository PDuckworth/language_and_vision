import numpy as np
from sklearn import mixture
import itertools
from sklearn.metrics.cluster import v_measure_score
import cv2
import pickle
import pulp
import sys
import colorsys
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import shuffle


colours = ['red','blue','green','yellow']
fig, ax = plt.subplots()

for c,i in enumerate(["faces","colours"]):
    dir2 = "/home/omari/Datasets/ECAI_dataset/"+i+"/"
    f_score = pickle.load(open(dir2+i+'_incremental.p',"rb"))
    x = np.arange(len(f_score))/float(10)*493
    ax.plot(x, f_score,'o-b',linewidth=2,c=colours[c],label=i)
    plt.ylim([0,1])
    # ax.plot(x, yp,'r')
    # ax.plot(x, yr,'g')
ax.grid(True, zorder=5)
plt.legend()
plt.show()
