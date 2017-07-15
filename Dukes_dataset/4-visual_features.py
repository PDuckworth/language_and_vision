import numpy as np
import math as m
from xml_functions import *
from nltk.tree import *
from nltk.tree import ParentedTree
import pickle

from sklearn import mixture
# import itertools
from sklearn.metrics.cluster import v_measure_score
from sklearn import metrics
from sklearn import svm
import matplotlib.pyplot as plt
#--------------------------------------------------------------------------------------------------------#

def _read_pickle(scene):
    pkl_file = '/home/omari/Datasets/Dukes_modified/scenes/'+str(scene)+'_layout.p'
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
            actions = ['discard'] ## lower ?!?!?!?
        elif x_O[0]!=x_O[1] or y_O[0]!=y_O[1] or z_O[0]!=z_O[1]:
            actions = ['approach,grasp,lift','discard','approach,grasp,lift,move,discard,depart']
    else:
        actions = [] #'nothing'
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

    tree = {}
    if actions == ['approach,grasp,lift']:
        tree['NLTK'] = "(V (Action "+actions[0]+") (Entity id_"+str(mov_obj)+"))"
        tree['py'] = {}
        tree['py']['A'] = actions[0]
        tree['py']['E'] = mov_obj
    elif actions == ['discard']:
        tree['NLTK'] = "(V (Action "+actions[0]+") (Entity id_"+str(mov_obj)+"))"
        tree['py'] = {}
        tree['py']['A'] = actions[0]
        tree['py']['E'] = mov_obj
        # tree['py']['D'] = [x[1],y[1],z[1]]
    elif actions == ['approach,grasp,lift','discard','approach,grasp,lift,move,discard,depart']:
        tree['NLTK'] = "(V (Action "+actions[2]+") (Entity id_"+str(mov_obj)+") (Destination "+str(x[1])+","+str(y[1])+","+str(z[1])+"))"
        tree['py'] = {}
        tree['py']['A'] = actions[2]
        tree['py']['E'] = mov_obj
        tree['py']['D'] = [x[1],y[1],z[1]]
    elif actions == ['nothing']:
        tree['NLTK'] = "(V (Action "+actions[0]+"))"
        tree['py'] = {}
        tree['py']['A'] = actions[0]
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

def _get_locations2(positions):
    locations = []
    mov_obj = None
    for obj in positions:
        if obj != 'gripper':
            x = positions[obj]['x'][1]
            y = positions[obj]['y'][1]

            if [x,y] not in locations:
                locations.append([x,y])
    # print locations

    return locations

def _get_colors(positions):
    colors = []
    for obj in positions:
        if obj != 'gripper':
            color = positions[obj]['F_HSV']
            for c in color.split('-'):
                if c not in colors:
                    colors.append(c)
    return colors

def _get_shapes(positions):
    shapes = []
    for obj in positions:
        if obj != 'gripper':
            shape = positions[obj]['F_SHAPE']
            for s in shape.split('-'):
                if s not in shapes:
                    shapes.append(s)

    # groups = {}
    # for obj in positions:
    #     if obj != 'gripper':
    #         x=positions[obj]['x'][0]
    #         y=positions[obj]['y'][0]
    #         if positions[obj]['F_SHAPE'] in ['cube','cylinder']:
    #             if (x,y) not in groups:
    #                 groups[(x,y)]=1
    #             else:
    #                 groups[(x,y)]+=1
    # for i in groups:
    #     if groups[i]>1:
    #         shapes.append('tower')
    #         break
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

def _func_directions(dx,dy,dz):
        dx = float(dx)
        dy = float(dy)
        dz = float(dz)
        max = np.max(np.abs([dx,dy,dz]))
        if np.abs(dx)/max < .5:
            dx = 0
        else:
            dx = np.sign(dx)

        if np.abs(dy)/max < .5:
            dy = 0
        else:
            dy = np.sign(dy)

        if np.abs(dz)/max < .5:
            dz = 0
        else:
            dz = np.sign(dz)
        return dx,dy,dz

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
                # d = cart2sph(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                d = _func_directions(x1[0]-x2[0],y1[0]-y2[0],z1[0]-z2[0])
                if d not in directions:
                    directions.append(d)
                # d = cart2sph(x1[1]-x2[1],y1[1]-y2[1],z1[1]-z2[1])
                d = _func_directions(x1[1]-x2[1],y1[1]-y2[1],z1[1]-z2[1])
                if d not in directions:
                    directions.append(d)
    return directions

def _get_temporal(v):
    temporal = []
    if len(v)>1:
        temporal = ['meets']
    return temporal

def _cluster_data(X, GT):
    best_v = 0
    lowest_bic = 10000000000
    for i in range(5):
        print '#####',i
        n_components_range = range(5, 10)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        lowest_bic = np.infty
        for cv_type in cv_types:
            for n_components in n_components_range:
                gmm = mixture.GaussianMixture(n_components=n_components,covariance_type=cv_type)
                gmm.fit(X)
                Y_ = gmm.predict(X)
                ######################################
                bic = gmm.bic(X)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    final_Y_ = Y_
                ######################################
                # Y_ = gmm.predict(X)
                # # print GT
                # # print Y_
                # v_meas = v_measure_score(GT, Y_)
                # if v_meas > best_v:
                #     best_v = v_meas
                #     final_clf = gmm
                #     print best_v
                #     final_Y_ = Y_

    _print_results(GT,final_Y_,best_gmm)

def _append_data(data, X_, unique_, GT_, mean, sigma):
    for i in data:
        if i not in unique_:
            unique_.append(i)
        # print i,len(i)
        d = unique_.index(i)+ np.random.normal(mean, sigma, 1)
        if X_ == []:
            X_ = [d]
        else:
            X_ = np.vstack((X_,d))
        GT_.append(i)
    return X_, unique_, GT_

def _append_data2(data, X_, unique_, GT_, mean, sigma):
    for i in data:
        if i not in unique_:
            unique_.append(i)
        # print i,len(i)
        d = i + np.random.multivariate_normal(mean, sigma, 1)[0]
        # X.append(d[0])
        # Y.append(d[1])
        if X_ == []:
            X_ = [d]
        else:
            X_ = np.vstack((X_,d))
        GT_.append(unique_.index(i))
    return X_, unique_, GT_

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    lists = []
    for i in range(n):
        list1 = np.arange( i*l/n+1 , (i+1)*l/n+1 )
        lists.append(list1)
    return lists

def _print_results(GT,Y_,best_gmm):
    #print v_measure_score(self.GT, self.Y_)
    true_labels = GT
    pred_labels = Y_
    print "\n dataset unique labels:", len(set(true_labels))
    print "number of clusters:", len(best_gmm.means_)
    print("Mutual Information: %.2f" % metrics.mutual_info_score(true_labels, pred_labels))
    print("Adjusted Mutual Information: %0.2f" % metrics.normalized_mutual_info_score(true_labels, pred_labels))
    print("Homogeneity: %0.2f" % metrics.homogeneity_score(true_labels, pred_labels))
    print("Completeness: %0.2f" % metrics.completeness_score(true_labels, pred_labels))
    print("V-measure: %0.2f" % metrics.v_measure_score(true_labels, pred_labels))

##########################################################################
# save values for furhter analysis
##########################################################################
for scene in range(1,1001):
    print 'extracting feature from scene : ',scene
    pkl_file = '/home/omari/Datasets/Dukes_modified/learning/'+str(scene)+'_visual_features.p'
    VF = {}
    positions = _read_pickle(scene)
    VF['actions'] = _get_actions(positions)
    VF['locations'] = _get_locations(positions)
    VF['color'] = _get_colors(positions)
    VF['type'] = _get_shapes(positions)
    # VF['distances'] = _get_distances(positions)
    VF['relation'] = _get_directions(positions)
    # VF['temporal'] = _get_temporal(VF['actions'])
    trees = _get_trees(VF['actions'],positions)
    pickle.dump([VF,trees], open(pkl_file, 'wb'))

    # print positions


##########################################################################
# Clustering analysis
##########################################################################
four_folds = chunks(1000,4)

X_colours = []
GT_colours = []
unique_colours = []

X_shapes = []
GT_shapes = []
unique_shapes = []

X_locations = []
GT_locations = []
unique_locations = []

for test in range(1):
    for c,data in enumerate(four_folds):
        if c != test:
            for scene in data:
                pkl_file = '/home/omari/Datasets/Dukes_modified/learning/'+str(scene)+'_visual_features.p'
                positions = _read_pickle(scene)
                X_colours, unique_colours, GT_colours           = _append_data(_get_colors(positions), X_colours, unique_colours, GT_colours, 0, .3)
                X_shapes, unique_shapes, GT_shapes              = _append_data(_get_shapes(positions), X_shapes, unique_shapes, GT_shapes, 0, .3)
                X_locations, unique_locations, GT_locations     = _append_data2(_get_locations2(positions), X_locations, unique_locations, GT_locations, [0,0], [[.3, 0], [0, .3]])
    # print X_locations
    # print unique_locations
    # print GT_locations
    _cluster_data(X_colours, GT_colours)
    _cluster_data(X_shapes, GT_shapes)
    _cluster_data(X_locations, GT_locations)
    # for i,j in zip(X_colours, GT_colours):
    #     print i,j
    # plt.plot(X, Y, 'rx')
    # plt.show()
