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
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/'+str(scene)+'_linguistic_features.p'
    data = open(pkl_file, 'rb')
    lf = pickle.load(data)
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/'+str(scene)+'_visual_features.p'
    data = open(pkl_file, 'rb')
    [vf,tree] = pickle.load(data)
    return lf,vf


LF_dict = {}
VF_dict = {}
pos = 0
for scene in range(1,1001):
    # print 'generating dictionaries from scene : ',scene
    LF,VF = _read_pickle(scene)
    for f in VF:
        for v in VF[f]:
            if f+'_'+str(v) not in VF_dict:
                VF_dict[f+'_'+str(v)] = {}
                VF_dict[f+'_'+str(v)]['VF'] = v
                VF_dict[f+'_'+str(v)]['count'] = 1
            else:
                VF_dict[f+'_'+str(v)]['count'] += 1
    for n in LF['n_grams']:
        if n == 'position':
            pos += 1
        if n not in LF_dict:
            LF_dict[n] = {}
            LF_dict[n]['count'] = 1
        else:
            LF_dict[n]['count'] += 1


# initiate the dictionaries
for n in LF_dict:
    for v in sorted(VF_dict.keys()):
        LF_dict[n][v] = 0

for scene in range(1,1001):
    # print 'filling dictionaries from scene : ',scene
    LF,VF = _read_pickle(scene)
    vf_in_this_scene = []
    for f in VF:
        for v in VF[f]:
            vf_in_this_scene.append(f+'_'+str(v))
    for n in LF['n_grams']:
        for v in vf_in_this_scene:
            LF_dict[n][v] += 1

rows = len(VF_dict.keys())
cols = len(LF_dict.keys())
cost_matrix = np.ones((rows,cols), dtype=np.float32)

for col,lexicon in enumerate(sorted(LF_dict.keys())):
    for row,v in enumerate(sorted(VF_dict.keys())):
        LF_dict[lexicon][v] = 1 - LF_dict[lexicon][v]/float(LF_dict[lexicon]['count'])
        cost_matrix[row,col] = LF_dict[lexicon][v]

# this is the hangarian algorithm
for row in range(len(VF_dict.keys())):
    cost_matrix[row,:]-=np.min(cost_matrix[row,:])
for col in range(len(LF_dict.keys())):
    cost_matrix[:,col]-=np.min(cost_matrix[:,col])

alpha = .01
sorted_LF = sorted(LF_dict.keys())
sorted_VF = sorted(VF_dict.keys())
hypotheses_tags = {}
for val2,VF in enumerate(sorted_VF):
    for val1,LF in enumerate(sorted_LF):
        if cost_matrix[val2,val1] <= alpha:
            if LF not in hypotheses_tags:
                hypotheses_tags[LF] = {}
            hypotheses_tags[LF][VF] = cost_matrix[val2,val1]
            print VF,'---',LF,':',cost_matrix[val2,val1]


pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/tags.p'
pickle.dump([hypotheses_tags, VF_dict, LF_dict], open(pkl_file, 'wb'))
