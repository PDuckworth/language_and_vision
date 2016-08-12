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
