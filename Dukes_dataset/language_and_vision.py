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
#--------------------------------------------------------------------------------------------------------#

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

def _read_vf(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/'+str(scene)+'_visual_features.p'
    data = open(pkl_file, 'rb')
    vf,tree = pickle.load(data)
    return vf,tree

def _read_semantic_trees(scene):
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/'+str(scene)+'_semantic_grammar.p'
    data = open(pkl_file, 'rb')
    tree = pickle.load(data)
    return tree

def _is_valid_action_query(tree):
    action_valid = 0
    query_valid = 0     # make sure every query is valid
    for query in tree:
        actions = 0
        for item in query:
            if 'actions_' in item:
                actions+=1
        if actions==0:
            query_valid+=1
        if actions>0 and len(query)==actions:
            query_valid+=1
    if query_valid == len(query):
        action_valid=1
    return action_valid


def _is_valid_query(tree):
    action_valid = _is_valid_action_query(tree)
    # print tree
    if action_valid:
        print tree


def _validate(tree):
    valid = _is_valid_query(tree)

hypotheses_tags, VF_dict, LF_dict = _read_tags()
for scene in range(17,18):
    semantic_trees = {}
    print 'test grammar from scene : ',scene
    VF,Tree = _read_vf(scene)
    sentences = _read_sentences(scene)
    semantic_trees = _read_semantic_trees(scene)
    counter = 0
    for id in semantic_trees:
        print id
        for grammar in semantic_trees[id]:
            for semantic in semantic_trees[id][grammar]:
                tree = semantic_trees[id][grammar][semantic]
                _validate(tree)
            # counter += len(semantic_trees[id][t])
