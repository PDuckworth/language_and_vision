import numpy as np
import math as m
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
import pickle
import getpass
import operator
#--------------------------------------------------------------------------------------------------------#

def _read_pickle(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/scenes/'+str(scene)+'_linguistic_features.p'
    data = open(pkl_file, 'rb')
    lf = pickle.load(data)
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/scenes/'+str(scene)+'_visual_features.p'
    data = open(pkl_file, 'rb')
    vf = pickle.load(data)
    return lf,vf


LF_dict = {}
VF_dict = {}
for scene in range(1,1000):
    print 'generating dictionaries from scene : ',scene
    LF,VF = _read_pickle(scene)
    for f in VF:
        for v in VF[f]:
            if f+'_'+str(v) not in VF_dict:
                VF_dict[f+'_'+str(v)] = 1
            else:
                VF_dict[f+'_'+str(v)] += 1
    for n in LF['n_grams']:
        if n not in LF_dict:
            LF_dict[n] = {}
            LF_dict[n]['count'] = 1
        else:
            LF_dict[n]['count'] += 1

# initiate the dictionaries
for n in LF_dict:
    for v in sorted(VF_dict.keys()):
        LF_dict[n][v] = 0

for scene in range(1,1000):
    print 'filling dictionaries from scene : ',scene
    LF,VF = _read_pickle(scene)
    vf_in_this_scene = []
    for f in VF:
        for v in VF[f]:
            vf_in_this_scene.append(f+'_'+str(v))
    for n in LF['n_grams']:
        for v in vf_in_this_scene:
            LF_dict[n][v] += 1
    # for n in LF['n_grams']:
x = LF_dict['top right corner']
sorted_x = sorted(x.items(), key=operator.itemgetter(1))
print sorted_x

# for k in sorted(VF_dict.keys()):
#     print k,VF_dict[k]
