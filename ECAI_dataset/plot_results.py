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


colours = ['red','blue','green','purple']
markers = ["o","^","*","s"]
fig, ax = plt.subplots()

for c,i in enumerate(["faces","colours","objects","actions"]):
# for c,i in enumerate(["faces","colours","objects"]):
    dir2 = "/home/omari/Datasets/ECAI_dataset/"+i+"/"
    f_score = []
    f = pickle.load(open(dir2+i+'_incremental.p',"rb"))
    for f1 in f:
        f_score.append(f1)

    x = np.arange(len(f_score))/float(5)*499
    print x
    ax.plot(x, f_score,markers[c]+'-b',linewidth=2,markersize=14,c=colours[c],label=i)
plt.xlim([-50,450])
plt.xticks([00,100,200,300,400], ['5-Apr','6-Apr','7-Apr','8-Apr','11-Apr'], fontsize=20)
plt.yticks([0,.1,.200,.300,.400,.5,.6], ['','0.1','0.2','0.3','0.4','0.5','0.6'], fontsize=20)
plt.ylabel("f1-score", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(True, zorder=5)
plt.legend(loc='lower right', fontsize=23)
plt.show()
