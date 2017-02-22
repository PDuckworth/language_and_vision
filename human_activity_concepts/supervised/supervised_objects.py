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
import random


def supervised_svm((ground_truth, vectors)):

    merged = zip(vectors, ground_truth)
    random.shuffle(merged)

    pred_labels = []
    true_labels = []

    num_folds = 4
    subset_size = len(merged)/num_folds

    for i in range(num_folds):
        train = merged[:i*subset_size] + merged[(i+1)*subset_size:]
        test = merged[i*subset_size:][:subset_size]

        X_train = [x for (x,y) in train]
        Y_train = [y for (x,y) in train]

        X_test = [x for (x,y) in test]
        Y_test = [y for (x,y) in test]

        C = 1.0
        # import pdb; pdb.set_trace()
        svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
        # svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, Y_train)

        for cnt, j in enumerate(X_test):
            label = svc.predict(j.reshape(1, len(j)))[0]
            pred_labels.append(label)

        true_labels.extend(Y_test)
        import pdb; pdb.set_trace()

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
    for feature in ["objects_supervised.p"]:

        print "\nFEATURE SPACE: %s" % feature

        with open("objects_supervised.p", "rb") as f:
            vectors = pickle.load(f)

        with open("object_ground_truth.p", "rb") as f:
            ground_truth = pickle.load(f)

        supervised_svm((ground_truth, labels))



        # num_of_folds = 4
        # merged = zip(data[1], data[0])
        # foldsize = int(np.floor(len(merged) / float(num_of_folds)))
        #
        # train = merged[:-foldsize]
        # test = merged[-foldsize:]
        #
        # X_train = [x for (x,y) in train]
        # Y_train = [y for (x,y) in train]
        #
        # X_test = [x for (x,y) in test]
        # Y_test = [y for (x,y) in test]
        #
        # C = 1.0
        # svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
        # # svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
        #
        # pred_labels = []
        # for cnt, i in enumerate(X_test):
        #     label = svc.predict(i.reshape(1, len(i)))[0]
        #     pred_labels.append(label)
        #
        # true_labels = Y_test
        #
        # print "\n supervised number of clusters:", len(set(true_labels))
        # print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, pred_labels))
        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, pred_labels))
        # print("Completeness: %0.3f" % metrics.completeness_score(true_labels, pred_labels))
        # print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(true_labels, pred_labels))
        # print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(true_labels, pred_labels))
