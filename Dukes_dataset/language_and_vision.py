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

#---------------------------------------------------------------------------#
def _read_tags():
    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/learning/tags.p'
    data = open(pkl_file, 'rb')
    hypotheses_tags, VF_dict, LF_dict = pickle.load(data)
    return [hypotheses_tags, VF_dict, LF_dict]

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
def _is_valid_action_query(tree):
    action_valid = 0
    query_valid = 0     # make sure every query is valid
    # this makes sure that only one action exist in the tree
    actions = 0
    for query in tree:
        for item in query:
            if 'actions_' in item:
                actions+=1
                break

    if actions == 1:
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

#---------------------------------------------------------------------------#
def _get_all_entities(Entity):
    _Entity = []
    features = ['colors_','shapes_','locations_']
    Entities = {}
    Relations = {}

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
            if new_entity:
                count+=1
                new_entity = 0
            _Entity.append(word)

    features = ['directions_']
    count = 0
    new_relation = 0
    _Entity_Relation = []
    for word in _Entity:
        R = 0
        for f in features:
            if f in word:
                R = 1
        if R:
            if count not in Relations:
                Relations[count] = []
                _Entity_Relation.append('Relation_'+str(count))
                new_relation = 1
            Relations[count].append(word)
        else:
            if new_relation:
                count+=1
                new_relation = 0
            _Entity_Relation.append(word)
    return [_Entity_Relation, Entities, Relations]

#---------------------------------------------------------------------------#
def _is_valid_entity_query(tree):
    Action = []
    Entity = []
    tree_structure = {}
    for count,query in enumerate(tree):
        for item in query:
            action_item = 0
            if 'actions_' in item:
                if item not in Action:
                    action_item = 1
        if action_item:
            Action = query
            tree_structure['A'] = count
        if not action_item:
            Entity = query
            tree_structure['E'] = count
    Entity,Entities,Relations = _get_all_entities(Entity)
    return [tree_structure,Action,Entity,Entities,Relations]

#---------------------------------------------------------------------------#
def _is_valid_entity_destination_query(tree):
    Action = []
    Entity,E_Entities,E_Relations = [],[],[]
    Destination,D_Entities,D_Relations = [],[],[]

    tree_structure = {}
    for count,query in enumerate(tree):
        for item in query:
            action_item = 0
            if 'actions_' in item:
                if item not in Action:
                    action_item = 1
        if action_item:
            Action = query
            tree_structure['A'] = count
        if not action_item and count==1:
            Entity = query
            tree_structure['E'] = count
            Entity,E_Entities,E_Relations = _get_all_entities(Entity)
        if not action_item and count==2:
            Destination = query
            tree_structure['D'] = count
            Destination,D_Entities,D_Relations = _get_all_entities(Destination)
    return [tree_structure,Action,Entity,E_Entities,E_Relations,Destination,D_Entities,D_Relations]

#---------------------------------------------------------------------------#
def _is_valid_query(tree):
    tree_structure,Action,Entity,Entities,Relations,Destination,D_Entities,D_Relations = [],[],[],[],[],[],[],[]
    action_valid = _is_valid_action_query(tree)
    if action_valid:
        if len(tree) == 2:
            tree_structure,Action,Entity,Entities,Relations = _is_valid_entity_query(tree)
        if len(tree) == 3:
            tree_structure,Action,Entity,Entities,Relations,Destination,D_Entities,D_Relations = _is_valid_entity_destination_query(tree)
    return [action_valid,tree_structure,Action,Entity,Entities,Relations,Destination,D_Entities,D_Relations]

#---------------------------------------------------------------------------#
def _match_action_with_scene(Action,Scene,VF_dict):
    valid = 1
    for A in Action:
        if VF_dict[A]['VF']!=Scene:
            valid = 0
    return valid

#---------------------------------------------------------------------------#
def _get_object_ids(feature,value,layout):
    ids = []
    if feature=='shapes':
        for id in layout:
            if id!='gripper':
                if layout[id]['F_SHAPE']==value:
                    ids.append(id)
    if feature=='colors':
        for id in layout:
            if id!='gripper':
                if layout[id]['F_HSV']==value:
                    ids.append(id)
    return ids

#---------------------------------------------------------------------------#
def cart2sph(x,y,z):
    num = 90
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))*180/np.pi     # theta
    elev = int(elev/num)*num
    az = m.atan2(y,x)*180/np.pi                           # phi
    az = int(az/num)*num
    return int(elev), int(az)

#---------------------------------------------------------------------------#
def _match_Entity_with_scene(Entity,Entities,Relations,VF_dict,layout,scene):
    scene_ids = {}
    valid_entity = 0
    for id in Entities:
        scene_ids[id] = []
        for f in Entities[id]:
            feature = f.split('_')[0]
            value = f.split('_')[1]
            ids = _get_object_ids(feature,value,layout)
            if ids==[]:
                scene_ids[id] = []
                break
            if scene_ids[id] == []:
                scene_ids[id] = ids
            else:
                scene_ids[id] = list(set(scene_ids[id]).intersection(ids))
    if len(scene_ids.keys()) == 1 and len(Entity)==1:
        if len(scene_ids[0]) == 1:
            if scene_ids[0][0] == scene:
                valid_entity = 1
    if len(scene_ids.keys()) == 2:
        ids = []
        if len(Entity) == 3:
            for id0 in scene_ids[0]:
                x1 = layout[id0]['x']
                y1 = layout[id0]['y']
                z1 = layout[id0]['z']
                for id1 in scene_ids[1]:
                    x2 = layout[id1]['x']
                    y2 = layout[id1]['y']
                    z2 = layout[id1]['z']
                    if 'directions_' in Relations[0][0]:
                        d = cart2sph(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                        if d==VF_dict[Relations[0][0]]['VF']:
                            ids.append(id0)
            if len(ids)==1:
                if ids[0] == scene:
                    valid_entity = 1
    return valid_entity

#---------------------------------------------------------------------------#
def _match_Destination_with_scene(Destination,D_Entities,D_Relations,VF_dict,layout,scene):
    scene_ids = {}
    valid_destination = 0
    if len(D_Entities)==1 and len(D_Relations)==1:
        for id in D_Entities:
            scene_ids[id] = []
            for f in D_Entities[id]:
                feature = f.split('_')[0]
                value = f.split('_')[1]
                ids = _get_object_ids(feature,value,layout)
                if ids==[]:
                    scene_ids[id] = []
                    break
                if scene_ids[id] == []:
                    scene_ids[id] = ids
                else:
                    scene_ids[id] = list(set(scene_ids[id]).intersection(ids))
        if len(scene_ids[id])==1:
            x1 = scene[0]
            y1 = scene[1]
            z1 = scene[2]
            id1 = scene_ids[id][0]
            x2 = layout[id1]['x'][1]
            y2 = layout[id1]['y'][1]
            z2 = layout[id1]['z'][1]

            if 'directions_' in D_Relations[0][0]:
                d = cart2sph(x1-x2,y1-y2,z1-z2)
                if d==VF_dict[D_Relations[0][0]]['VF']:
                    valid_destination = 1
    return valid_destination

#---------------------------------------------------------------------------#
def _validate(tree, scene_tree, grammar, scene, id ,g):
    # print '##############################################################'
    # print grammar
    pass_flag=0
    valid,tree_structure,Action,Entity,Entities,Relations,Destination,D_Entities,D_Relations = _is_valid_query(tree)
    if valid:
        valid_action = _match_action_with_scene(Action,scene_tree['A'],VF_dict)
        if valid_action:
            if len(scene_tree)==2:
                valid_entity = _match_Entity_with_scene(Entity,Entities,Relations,VF_dict,layout,scene_tree['E'])
                if valid_entity:
                    results = {}
                    results['grammar'] = grammar
                    results['semantic'] = tree
                    results['tree_structure'] = tree_structure
                    results['entity'] = [Entity,Entities,Relations]
                    pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/matching/'+str(scene)+'_'+str(id)+'_'+str(g)+'_matched_tree.p'
                    pickle.dump(results, open(pkl_file, 'wb'))
                    pass_flag = 1
            if len(scene_tree)==3:
                valid_entity = _match_Entity_with_scene(Entity,Entities,Relations,VF_dict,layout,scene_tree['E'])
                if valid_entity:
                    valid_destination = _match_Destination_with_scene(Destination,D_Entities,D_Relations,VF_dict,layout,scene_tree['D'])
                    if valid_destination:
                        results = {}
                        results['grammar'] = grammar
                        results['semantic'] = tree
                        results['tree_structure'] = tree_structure
                        results['entity'] = [Entity,Entities,Relations]
                        results['destination'] = [Destination,D_Entities,D_Relations]
                        pkl_file = '/home/'+getpass.getuser()+'/Datasets_old/Dukes_modified/matching/'+str(scene)+'_'+str(id)+'_'+str(g)+'_matched_tree.p'
                        pickle.dump(results, open(pkl_file, 'wb'))
                        pass_flag = 1
    return pass_flag
#---------------------------------------------------------------------------#
hypotheses_tags, VF_dict, LF_dict = _read_tags()
counter = []
for scene in range(10,11):
    layout = _read_layout(scene)
    semantic_trees = {}
    print 'test grammar from scene : ',scene
    VF,scene_tree = _read_vf(scene)
    grammar_trees = _read_grammar_trees(scene)
    semantic_trees = _read_semantic_trees(scene)
    for id in semantic_trees:
        print id
        print grammar_trees[id]
        for g in semantic_trees[id]:
            for semantic in semantic_trees[id][g]:
                tree = semantic_trees[id][g][semantic]
                print tree
                pass_flag = _validate(tree, scene_tree['py'], grammar_trees[id][g],scene,id,g)
                if pass_flag and scene not in counter:
                    counter.append(scene)
print len(counter)
print counter
