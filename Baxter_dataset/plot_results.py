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
markers_size = [11,11,15,11]
fig, ax = plt.subplots()

for c,i in enumerate(["shapes","colours","distances"]):
# for c,i in enumerate(["faces","colours","objects"]):
    dir2 = "/home/omari/Datasets/Baxter_Dataset_final/features/language/"
    f_score = []
    f = pickle.load(open(dir2+i+'_incremental.p',"rb"))
    for f1 in f:
        f_score.append(f1)

    x = np.arange(len(f_score))/float(5)*499
    print x
    ax.plot(x, f_score,markers[c]+'-b',linewidth=2,markersize=markers_size[c],c=colours[c],label=i)
plt.xlim([-50,450])
plt.xticks([00,100,200,300,400], ['20%','40%','60%','80%','100%'], fontsize=20)
plt.yticks([0,.05,.1,.15,.2,.25,.3], ['0.0','0.05','0.1','0.15','0.2','0.25','0.3'], fontsize=20)
plt.ylabel("F1-score", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.grid(True, zorder=5)
plt.legend(loc='lower right', fontsize=23)
plt.show()
