import numpy as np
import math as m
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
import pickle
#--------------------------------------------------------------------------------------------------------#

def _read_pickle(scene):
    pkl_file = '/home/omari/Datasets_old/Dukes_modified/scenes/'+str(scene)+'_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences


def _find_n_grams(sentence):
    n_word = 3 ## length of n_grams
    w = sentence.split(' ')
    n_grams = []
    for i in range(len(w)):
        # if w[i]not in self.words[s]: self.words[s].append(w[i])
        for j in range(i+1,np.min([i+1+n_word,len(w)+1])):
            n_grams.append(' '.join(w[i:j]))
    return n_grams

def _get_n_grams(sentences):
    all_n_grams = []
    for id in sentences:
        n = _find_n_grams(sentences[id]['text'])
        for i in n:
            if i not in all_n_grams:
                all_n_grams.append(i)
    return all_n_grams


for scene in range(1,1001):
    print 'extracting feature from scene : ',scene
    pkl_file = '/home/omari/Datasets_old/Dukes_modified/scenes/'+str(scene)+'_linguistic_features.p'
    LF = {}
    sentences = _read_pickle(scene)
    LF['n_grams'] = _get_n_grams(sentences)
    # print LF['n_grams']
    pickle.dump(LF, open(pkl_file, 'wb'))
