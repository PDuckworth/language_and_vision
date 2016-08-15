import numpy as np
import math as m
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
import pickle
#--------------------------------------------------------------------------------------------------------#

def _read_pickle(scene):
    pkl_file = '/home/omari/Datasets_old/Dukes_modified/scenes/'+str(scene)+'_layout.p'
    data = open(pkl_file, 'rb')
    positions = pickle.load(data)
    return positions

def _get_actions(positions):
    actions = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x_O = positions[mov_obj]['x']
        y_O = positions[mov_obj]['y']
        z_O = positions[mov_obj]['z']

        x_R = positions['gripper']['x']
        y_R = positions['gripper']['y']
        z_R = positions['gripper']['z']

        # check if it's a pick up
        if x_O[1]==x_R[1] and y_O[1]==y_R[1] and z_O[1]==z_R[1]:
            actions = ['approach,grasp,lift']
        elif x_O[0]==x_R[0] and y_O[0]==y_R[0] and z_O[0]==z_R[0]:
            actions = ['move,discard,depart'] ## lower ?!?!?!?
        elif x_O[0]!=x_O[1] or y_O[0]!=y_O[1] or z_O[0]!=z_O[1]:
            actions = ['approach,grasp,lift','move,discard,depart','approach,grasp,lift,move,discard,depart']
        else:
            actions = []
            print positions[mov_obj]
            print positions['gripper']
    return actions

def _get_trees(actions,positions):
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break

    if mov_obj != None:
        x = positions[mov_obj]['x']
        y = positions[mov_obj]['y']
        z = positions[mov_obj]['z']

    tree = ""
    if actions == ['approach,grasp,lift']:
        tree = "(V (action "+actions[0]+") (Entity id_"+str(mov_obj)+"))"
    elif actions == ['move,discard,depart']:
        tree = "(V (action "+actions[0]+") (Entity id_"+str(mov_obj)+") (Location "+str(x[1])+","+str(y[1])+","+str(z[1])+")"
    elif actions == ['approach,grasp,lift','move,discard,depart','approach,grasp,lift,move,discard,depart']:
        tree = ""
    return tree

def _get_locations(positions):
    locations = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x = positions[mov_obj]['x']
        y = positions[mov_obj]['y']
        if x[0]<3 and y[0]<3:
            locations.append([0,0])
        if x[0]<3 and y[0]>4:
            locations.append([0,7])
        if x[0]>4 and y[0]<3:
            locations.append([7,0])
        if x[0]>4 and y[0]>4:
            locations.append([7,7])
        if x[0]>1 and x[0]<5 and y[0]>1 and y[0]<5:
            locations.append([3.5,3.5])

        if x[1]<3 and y[1]<3:
            locations.append([0,0])
        if x[1]<3 and y[1]>4:
            locations.append([0,7])
        if x[1]>4 and y[1]<3:
            locations.append([7,0])
        if x[1]>4 and y[1]>4:
            locations.append([7,7])
        if x[1]>1 and x[1]<5 and y[1]>1 and y[1]<5:
            locations.append([3.5,3.5])

    return locations

def _get_colors(positions):
    colors = []
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['F_HSV'] not in colors:
                colors.append(positions[obj]['F_HSV'])
    return colors

def _get_shapes(positions):
    shapes = []
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['F_SHAPE'] not in shapes:
                shapes.append(positions[obj]['F_SHAPE'])
    return shapes

def _get_distances(positions):
    distances = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x1 = positions[mov_obj]['x']
        y1 = positions[mov_obj]['y']
        # z1 = positions[mov_obj]['z']
        for obj in positions:
            if obj != 'gripper' and obj != mov_obj:
                x2 = positions[obj]['x']
                y2 = positions[obj]['y']
                d = [np.abs(x1[0]-x2[0]),np.abs(x1[1]-x2[1]),np.abs(y1[0]-y2[0]),np.abs(y1[1]-y2[1])]
                for i in d:
                    if i not in distances:
                        distances.append(i)
    return distances

def cart2sph(x,y,z):
    num = 90
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))*180/np.pi     # theta
    elev = int(elev/num)*num
    az = m.atan2(y,x)*180/np.pi                           # phi
    az = int(az/num)*num
    return int(elev), int(az)

def _get_directions(positions):
    #http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    directions = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            if positions[obj]['moving']:
                mov_obj = obj
                break
    if mov_obj != None:
        x1 = positions[mov_obj]['x']
        y1 = positions[mov_obj]['y']
        z1 = positions[mov_obj]['z']
        for obj in positions:
            if obj != 'gripper' and obj != mov_obj:
                x2 = positions[obj]['x']
                y2 = positions[obj]['y']
                z2 = positions[obj]['z']
                d = cart2sph(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                if d not in directions:
                    directions.append(d)
                d = cart2sph(x1[1]-x2[1],y1[1]-y2[1],z1[1]-z2[1])
                if d not in directions:
                    directions.append(d)
    return directions

for scene in range(1,1001):
    print 'extracting feature from scene : ',scene
    pkl_file = '/home/omari/Datasets_old/Dukes_modified/learning/'+str(scene)+'_visual_features.p'
    VF = {}
    positions = _read_pickle(scene)
    VF['actions'] = _get_actions(positions)
    VF['locations'] = _get_locations(positions)
    VF['colors'] = _get_colors(positions)
    VF['shapes'] = _get_shapes(positions)
    # VF['distances'] = _get_distances(positions)
    VF['directions'] = _get_directions(positions)
    pickle.dump(VF, open(pkl_file, 'wb'))

    tree_file = '/home/omari/Datasets_old/Dukes_modified/learning/'+str(scene)+'_video_tree.p'
    trees = _get_trees(VF['actions'],positions)
    print trees
