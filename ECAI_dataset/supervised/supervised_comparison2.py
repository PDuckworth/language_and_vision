#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import time
import scipy
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import getpass
from sklearn import svm, metrics



def supervised_svm((ground_truth, vectors)):

    merged = zip(vectors, ground_truth)
    pred_labels = []
    true_labels = []

    num_folds = 4
    subset_size = len(vectors)/num_folds

    for i in range(num_folds):
        train = merged[:i*subset_size] + merged[(i+1)*subset_size:]
        test = merged[i*subset_size:][:subset_size]

        X_train = [x for (x,y) in train]
        Y_train = [y for (x,y) in train]

        X_test = [x for (x,y) in test]
        Y_test = [y for (x,y) in test]

        C = 1.0
        svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
        # svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)

        for cnt, i in enumerate(X_test):
            label = svc.predict(i.reshape(1, len(i)))[0]
            pred_labels.append(label)
        true_labels.extend(Y_test)


    print "\n supervised number of clusters:", len(set(true_labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, pred_labels))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, pred_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(true_labels, pred_labels))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(true_labels, pred_labels))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(true_labels, pred_labels))

    return



if __name__ == "__main__":
    """	Read the feature vector and labels files
    """

    #read the data
    path = '/home/' + getpass.getuser() + '/Dropbox/Programming/topics_to_language/supervised'

    for feature in ["face_supervised.p", "colour_supervised.p"]:
        with open(os.path.join(path, feature), "rb")  as f:
            data = pickle.load(f)
        supervised_svm(data)










    #create a feature vector and label per instance
