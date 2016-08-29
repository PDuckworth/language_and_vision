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
def _read_RCL_tree(id):
    pkl_file = '/home/omari/Datasets_old/Dukes_modified/RCL-trees/'+str(id)+'_tree.p'
    data = open(pkl_file, 'rb')
    RCL_tree = pickle.load(data)
    return RCL_tree

#---------------------------------------------------------------------------#
def _read_tags():
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/tags.p'
    data = open(pkl_file, 'rb')
    hypotheses_tags, VF_dict, LF_dict = pickle.load(data)
    return [hypotheses_tags, VF_dict, LF_dict]

#---------------------------------------------------------------------------#
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

#---------------------------------------------------------------------------#
def _read_passed_tags():
    pkl_file = '/home/omari/Datasets_old/Dukes_modified/matching/Passed_tags1.p'
    data = open(pkl_file, 'rb')
    Matching,Matching_VF,passed_scenes,passed_sentences = pickle.load(data)
    # print Matching,Matching_VF,passed_scenes,passed_ids
    return [Matching,Matching_VF,passed_scenes,passed_sentences]

#---------------------------------------------------------------------------#
def _is_yuk(sentence):
    yuk=0
    yuk_words = [' and',' closest',' near',' far',' nearest', ' edge', ' corner' ,' side', ' leftmost', 'rightmost' ,'1','2','3','4','5','row','two','one','three','five', 'four', 'single', 'position', 'grid','right','lift','box','left','location','exactly','lower','that','next','lowest','opposite',' it',' to ']
    for word in yuk_words:
        if word in sentence:
            yuk=1
            break
    return yuk

def _read_tree(id):
    pkl_file = '/home/omari/Datasets_old/Dukes_modified/matching/'+str(id)+'.p'
    data = open(pkl_file, 'rb')
    results = pickle.load(data)
    return results

def _create_simple_entity(categries,words):
    sub_trees = []
    for id in categries:
        for cat,word in zip(categries[id],words):
            sub_trees.append(Tree(cat.split('_')[0]+':',[word]))
    return sub_trees
    # print words

def _get_entity(results):
    struct = results['tree_structure']
    grammar = results['grammar']
    entity = results['entity']
    print entity
    if len(entity[0])==1:
        Ent = _create_simple_entity(entity[1],grammar[struct['E']])
    if len(entity[0])==3:
        print 'FIXXXXXX MEEEEEE !!!!! help help'
        Ent = ['fix me']
    return Ent

def _get_action(results):
    struct = results['tree_structure']
    grammar = results['grammar']
    action = (' ').join(grammar[struct['A']])
    return [action]

# def _get_relation(results):
#     struct = results['tree_structure']
#     grammar = results['grammar']
#     action = (' ').join(grammar[struct['A']])
#     return [action]

def _get_destination(results):
    struct = results['tree_structure']
    grammar = results['grammar']
    destination = results['destination']
    words = grammar[struct['D']]
    count = 0
    rel_words = []
    for r in destination[2]:
        for rel in destination[2][r]:
            rel_words.append(words[count])
            count+=1
    R=Tree('relation:', [(' ').join(rel_words)])
    E=Tree('entity:', _create_simple_entity(destination[1],words[count:]))
    dest = Tree('spatial-relation:',[R,E])
    return [dest]
#---------------------------------------------------------------------------#
Matching,Matching_VF,passed_scenes,passed_sentences = _read_passed_tags()
sentences_to_test = {}
counter = 0
counter2= 0
bad_trees = [14588,23958,10646,25409,25625,14427,23982,16360,22369,23928,16792,18058,25013,9323,26997,25565,14412,16159,26955,4028,9207,18582,25100,25058,23428,23985,12027,25653,14624,14423, 25682,12515,13775,4073,10186,13046,25622,26283,23217,12453,23955,23970,23756,23898,14789,25477,9418,2541,23738,24170]
for scene in range(1,1001):
    print '###',scene
    sentences = _read_sentences(scene)
    for id in sentences:
        if id not in bad_trees:
            if not _is_yuk(sentences[id]['text']):
                sentences_to_test[id] = sentences[id]
                if id in passed_sentences:
                    RCL_tree = _read_RCL_tree(id)
                    print RCL_tree
                    # print 'sentence:',sentences[id]['text']
                    results = _read_tree(id)
                    struct = results['tree_structure']
                    A=Tree('action:', _get_action(results))
                    E=Tree('entity:', _get_entity(results))
                    if len(struct)==2:
                        tree = Tree('event:', [A, E])
                    if len(struct)==3:
                        D=Tree('destination:', _get_destination(results))
                        tree = Tree('event:', [A, E, D])
                    # print tree
                    if tree==RCL_tree:
                        counter+=1
                    else:
                        print tree
                        print RCL_tree
                        counter2+=1
print counter
print counter2
