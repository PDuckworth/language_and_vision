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
    if query_valid == len(tree):
        action_valid=1
    return action_valid

def _get_all_entities(Entity):
    Entity = ['colors_magenta', 'shapes_sphere', 'Dir_','Dir_','colors_red', 'shapes_sphere']
    _Entity = []
    features = ['colors_','shapes_','locations_']
    Entities = {}

    count = 0
    new_entity = 0
    for word in Entity:
        E = 0
        for f in features:
            if f in word:
                E = 1
        if E:
            if count not in Entities:
                Entities[count] = []
                _Entity.append('Entity_'+str(count))
                new_entity = 1
            Entities[count].append(word)
        else:
            # _Entity.append('Entity_'+str(count))
            if new_entity:
                count+=1
                new_entity = 0
            _Entity.append(word)
    print _Entity
    print Entities




def _is_valid_entity_query(tree):
    Action = []
    Entity = []
    for query in tree:
        for item in query:
            action_item = 0
            if 'actions_' in item:
                if item not in Action:
                    action_item = 1
        if action_item:
            Action = query
        if not action_item:
            Entity = query
    print 'A >>>',Action
    print 'E >>>',Entity
    Entity = _get_all_entities(Entity)


def _is_valid_entity_destination_query(tree):
    print '>>>>',tree

def _is_valid_query(tree):
    action_valid = _is_valid_action_query(tree)
    if action_valid:
        if len(tree) == 2:
            Entity_valid = _is_valid_entity_query(tree)
        if len(tree) == 3:
            Entity_valid = _is_valid_entity_destination_query(tree)
        # print tree

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
