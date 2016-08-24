import numpy as np
import math as m
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
import pickle
import getpass
import operator
import itertools
from copy import deepcopy
from nltk.tree import *
from nltk.tree import ParentedTree

#---------------------------------------------------------------------------#
def _read_stop_wrods():
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/idf_FW_linguistic_features.p'
    data = open(pkl_file, 'rb')
    stop = pickle.load(data)
    return stop

#---------------------------------------------------------------------------#
def _read_tags():
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/tags.p'
    data = open(pkl_file, 'rb')
    hypotheses_tags, VF_dict, LF_dict = pickle.load(data)
    return [hypotheses_tags, VF_dict, LF_dict]

def _read_sentences(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/scenes/'+str(scene)+'_sentences.p'
    data = open(pkl_file, 'rb')
    sentences = pickle.load(data)
    return sentences

#---------------------------------------------------------------------------#
def _read_vf(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/'+str(scene)+'_visual_features.p'
    data = open(pkl_file, 'rb')
    vf,tree = pickle.load(data)
    return vf,tree

#---------------------------------------------------------------------------#
def _read_semantic_trees(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/'+str(scene)+'_semantic_grammar.p'
    data = open(pkl_file, 'rb')
    tree = pickle.load(data)
    return tree

#---------------------------------------------------------------------------#
def _read_layout(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/scenes/'+str(scene)+'_layout.p'
    data = open(pkl_file, 'rb')
    layout = pickle.load(data)
    return layout

#---------------------------------------------------------------------------#
def _read_grammar_trees(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/'+str(scene)+'_grammar.p'
    data = open(pkl_file, 'rb')
    tree = pickle.load(data)
    return tree

def _read_passed_tags():
    pkl_file = '/home/omari/Datasets_old/Dukes_modified/matching/Passed_tags1.p'
    data = open(pkl_file, 'rb')
    Matching,Matching_VF,passed_scenes,passed_sentences = pickle.load(data)
    # print Matching,Matching_VF,passed_scenes,passed_ids
    return [Matching,Matching_VF,passed_scenes,passed_sentences]


#---------------------------------------------------------------------------#
counter1 = 0
counter2 = 0
counter3 = 0
yuk_words = [' and',' closest',' near',' far',' nearest', ' edge', ' corner' ,' side', ' leftmost', 'rightmost' ,'1','2','3','4','5','row','two','one','three','five', 'four', 'single', 'position', 'grid','right','lift','box','left','location','exactly','lower','that','next','lowest','opposite',' it',' to ']

Matching,Matching_VF,passed_scenes,passed_sentences = _read_passed_tags()
sentences_to_test = {}
for scene in range(1,1001):
    print '###',scene
    sentences = _read_sentences(scene)
    for id in sentences:
        counter1+=1
        yuk=0
        for word in yuk_words:
            if word in sentences[id]['text']:
                yuk=1
        if not yuk:
            sentences_to_test[id] = sentences[id]
            if id in passed_sentences:
                print 'pass',sentences[id]['text']
                counter3+=1
            else:
                print 'NO',sentences[id]['text']
            counter2+=1
    print '---------'
print 'all sentences =',counter1
print 'ok sentences =',counter2
print 'passed sentences =',counter3


pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/experiment/sentences.p'
pickle.dump(sentences_to_test, open(pkl_file, 'wb'))

file1 = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/experiment/sentences.txt'
F = open(file1, 'w')
for id in sentences_to_test:
    F.write(sentences_to_test[id]['text']+'\n')
F.close()



Data = read_data()
tokens = [str(i) for i in range(150)]
f = open('/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/experiment/tags.txt', 'w')
for id in sentences_to_test:
    tree = ParentedTree.fromstring(sentences_to_test[id]['RCL'])
    sentence = []
    for p in tree.pos():
        if str(p[0]) not in tokens:
            # print (p[0],p[1].split(':')[0])
            sentence.append(p[1].split(':')[0])
    S = ' '.join(sentence)
    f.write(S+'\n')
f.close()
