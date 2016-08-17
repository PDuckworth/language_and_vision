import numpy as np
import math as m
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
import pickle
import getpass
import operator
#--------------------------------------------------------------------------------------------------------#
def _read_sentences(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/scenes/'+str(scene)+'_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences

def _read_tfidf_words():
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/idf_FW_linguistic_features.p'
    data = open(pkl_file, 'rb')
    tfidf = pickle.load(data)
    return tfidf

def _read_vf(scene):
    # pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/'+str(scene)+'_linguistic_features.p'
    # data = open(pkl_file, 'rb')
    # lf = pickle.load(data)
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/'+str(scene)+'_visual_features.p'
    data = open(pkl_file, 'rb')
    vf,tree = pickle.load(data)
    return vf,tree

pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/tags.p'
data = open(pkl_file, 'rb')
hypotheses_tags, VF_dict, LF_dict = pickle.load(data)
# this is why I can't have nice things
# total = 1
# for word in hypotheses_tags:
#     total*=len(hypotheses_tags[word].keys())+1
# print '>>>>>>>>>>>>>>>>>>',total
# In this dataset, we have 114 unique words, were each word has between 2 and 4 potential visual tags. If we compute the combinations of these words tags it's is 1.7 quattuordecillion 1.7*10^45, which gives the reader an idea of how massive the search space is, and that it can't be the case where we keep track of all combinations. Therefore we take a different approach, were we do the learning incrementally. The system analyse each snetnece seperatly, and record

tfidf_words = _read_tfidf_words()

for scene in range(2,3):
    print 'generating grammar from scene : ',scene
    VF,Tree = _read_vf(scene)
    print Tree
    sentences = _read_sentences(scene)
    for id in sentences:
        if id != 21232: continue
        S = sentences[id]['text'].split(' ')
        for word in tfidf_words:
            S = filter(lambda a: a != word, S)
        for word in S:
            print word,hypotheses_tags[word].keys()
        # break
        # S = (' ').join(S)
        print '---------------'
