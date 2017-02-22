#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import cPickle as pickle
import math
import numpy as np
import getpass
from onlineldavb import OnlineLDA
import pdb
import lda
import copy
import pyLDAvis
from sklearn import metrics
import matplotlib.pyplot as plt

def term_frequency_mat(codebook_lengh, wordids, wordcnts):
    term_freq = []
    for counter, (ids, cnts) in enumerate(zip(wordids, wordcnts)):
        # print ids, cnts
        vec = np.array([0] * codebook_lengh)
        for i, cnt in zip(ids, cnts):
            # print i, cnt
            vec[i] = cnt
        # print "vec: ", vec
        term_freq.append(vec)
        # print "tf: ", term_freq
    feature_space = np.vstack(term_freq)

    # for e, i in enumerate(term_freq):
    #     print e, sum(i)
    #     if sum(i) == 0:
    #         print ">", wordids[e], wordcnts[e]
    # pdb.set_trace()
    return feature_space

def get_dic_codebook(code_book, graphlets, create_graphlet_images=False):
    """code book already contains stringed hashes. """
    dictionary_codebook = dict(zip(code_book, graphlets))

    # dictionary_codebook = {}
    # for hash, graph in zip(code_book, graphlets):
    #     dictionary_codebook[g_name] = graph
    if create_graphlet_images:
        image_path = '/home/' + getpass.getuser() + '/Dropbox/Programming/topics_to_language/LDAvis_images'
        create_codebook_images(dictionary_codebook, image_path)
    return dictionary_codebook

def object_nodes(graph):
    object_nodes = []
    num_of_eps = 0
    for node in graph.vs():
        if node['node_type'] == 'object':
            if node['name'] not in ["hand", "torso"]:
                object_nodes.append(node['name'])
        if node['node_type'] == 'spatial_relation':
            num_of_eps+=1
    return object_nodes, num_of_eps

def print_results(true_labels, pred_labels, num_clusters):
    (h, c, v) =  metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)

    print "#Topics=%s (%s). v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. Acc: %0.3f" \
      % (num_clusters, len(pred_labels), v, h, c,
        metrics.mutual_info_score(true_labels, pred_labels),
        metrics.normalized_mutual_info_score(true_labels, pred_labels),
        metrics.accuracy_score(true_labels, pred_labels))

    row_inds = {}
    set_of_true_labs = list(set(true_labels))
    for cnt, i in enumerate(set_of_true_labs):
        row_inds[i] = cnt
    print "row inds",  row_inds

    res = {}
    mat = np.zeros( (len(set_of_true_labs), num_clusters))

    for i, j in zip(true_labels, pred_labels):
        row_ind = row_inds[i]
        mat[row_ind][j] +=1
        try:
            res[i].append(j)
        except:
            res[i] = [j]

    for cnt, i in enumerate(mat):
        print "label: %s: %s" % (set_of_true_labs[cnt], i)

    # pdb.set_trace()

    # Find assignments:
    for true, preds in res.items():
        print true, max(set(preds), key=preds.count)

    norm_mat = []
    for i in mat:
        norm_mat.append([float(j)/float( sum(i, 0) ) for j in i ])

    f, ax = plt.subplots(nrows=1, ncols=1)
    f.suptitle("Topic Confusion Marix: using segmented clips")
    plt.xlabel('Learnt Topics')
    plt.ylabel('Ground Truth Labels')
    plt.setp((ax), xticks=range(len(mat[0])), yticks=xrange(len(set_of_true_labs)))
    plt.yticks(xrange(len(set_of_true_labs)), set_of_true_labs, color='red')
    ax.imshow(mat, interpolation='nearest')

    # width, height = reordered.shape
    # for x in xrange(width):
    #     for y in xrange(height):
    #         ax2.annotate("%0.2f" % reordered[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center')

    # plt.show()

    return (num_clusters, len(pred_labels),
       metrics.v_measure_score(true_labels, pred_labels),
       metrics.homogeneity_score(true_labels, pred_labels),
       metrics.completeness_score(true_labels, pred_labels),
       metrics.mutual_info_score(true_labels, pred_labels),
       metrics.normalized_mutual_info_score(true_labels, pred_labels),
       metrics.accuracy_score(true_labels, pred_labels),
       mat, set_of_true_labs)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Usage: please provide a QSR run folder number."
        sys.exit(1)
    else:
        run = sys.argv[1]

    # ****************************************************************************************************
    # parameters
    # ****************************************************************************************************
    n_iters = 1000
    n_topics = 11
    create_graphlet_images = False
    dirichlet_params = (0.5, 0.01)
    _lambda = 0.5

    using_language = False
    # ****************************************************************************************************
    # load word counts
    # ****************************************************************************************************
    uuids, labels, wordids, wordcts = [], [], [], []
    directory = '/home/'+getpass.getuser()+'/Datasets/ECAI_Data/dataset_segmented_15_12_16/QSR_path/run_%s' % run
    print "directory: ", directory
    with open(directory+"/codebook_data.p", 'r') as f:
        loaded_data = pickle.load(f)

    (global_codebook, all_graphlets, uuids, labels) = loaded_data
    # print ">>", len(global_codebook), len(all_graphlets), len(uuids), len(labels)
    vocab = loaded_data[0]
    codebook_lengh = len(vocab)
    num_of_vids = len(os.listdir(directory))

    ordered_labels, ordered_vid_names = [], []
    for task in xrange(1,num_of_vids+1):
        if task in [196, 211]: continue
        video = "vid%s.p" % task
        d_video = os.path.join(directory, video)

        if not os.path.isfile(d_video): continue
        with open(d_video, 'r') as f:
            try:
                (uuid, label, ids, histogram) = pickle.load(f)
            except:
                "failed to load properly: \n %s" % (task)

        wordids.append(ids)
        wordcts.append(histogram)
        ordered_labels.append(label)
        ordered_vid_names.append(task)
    print "#videos: ", len(wordids), len(wordcts)

    K = 11
    D = 500
    alpha, eta = 0.1, 0.03
    tau0 = 10
    kappa = 0.7
    batchsize = 10
    num_iters = int(len(wordids)/float(batchsize))
    class_thresh = 0.3

    true_labels, pred_labels = [], []
    olda = OnlineLDA(global_codebook, K, D, alpha, eta, tau0, kappa, 0)

    # import random
    # a = ordered_labels
    # b = ordered_vid_names
    # c = list(zip(a, b))
    # random.shuffle(c)
    # ordered_labels, ordered_vid_names = zip(*c)

    for iteration in range(0, num_iters):
        # iteration = num_iters - iteration

        # print "it: %s. " %iteration #start: %s. end: %s" % (iteration, iteration*batchsize, (iteration+1)*batchsize)
        ids = wordids[iteration*batchsize:(iteration+1)*batchsize]
        cts = wordcts[iteration*batchsize:(iteration+1)*batchsize]

        labels = ordered_labels[iteration*batchsize:(iteration+1)*batchsize]
        vids = ordered_vid_names[iteration*batchsize:(iteration+1)*batchsize]

        (gamma, bound) = olda.update_lambda(ids, cts)
        print ">>", bound, len(wordids), D, sum(map(sum, cts)), olda._rhot

        thresholded_true, thresholded_pred = [], []
        for n, gam in enumerate(gamma):
            gam = gam / float(sum(gam))

            if max(gam) > class_thresh:
                # thresholded_true.append(ordered_labels[n])
                thresholded_true.append(labels[n])
                thresholded_pred.append(np.argmax(gam))

        # true_labels.extend(labels)
        # pred_labels.extend([np.argmax(i) for i in gamma])

        true_labels.extend(thresholded_true)
        pred_labels.extend(thresholded_pred)

        word_counter = 0
        for i in cts:
            word_counter+=sum(i)

        perwordbound = bound * len(wordids) / (D * word_counter)
        print 'iter: %s:  rho_t = %f,  held-out per-word perplexity estimate = %f. LDA - Done\n' % \
            (iteration, olda._rhot, np.exp(-perwordbound))

        if (iteration % 1 == 0):
            # import pdb; pdb.set_trace()
            if not os.path.exists(os.path.join(directory, 'TopicData')): os.makedirs(os.path.join(directory, 'TopicData'))
            np.savetxt(directory + '/TopicData/lambda-%d.dat' % iteration, olda._lambda)
            np.savetxt(directory + '/TopicData/gamma-%d.dat' % iteration, gamma)

            np.savetxt(directory + '/TopicData/vids-%d.dat' % iteration, vids, delimiter=" ", fmt="%s")
            np.savetxt(directory + '/TopicData/labels-%d.dat' % iteration, labels, delimiter=" ", fmt="%s")

    # # ****************************************************************************************************
    # # find the probability of each word from the term frequency matrix
    # # ****************************************************************************************************
    # sum_of_all_words = term_freq.sum()
    # sum_of_text_words = term_freq.T[language_indices].sum()
    # sum_of_graphlet_words = sum_of_all_words - sum_of_text_words
    #
    # probability_of_words = term_freq.sum(axis = 0) / float(sum_of_graphlet_words)
    # for ind in language_indices:
    #     probability_of_words[ind] = term_freq.T[ind].sum() / float(sum_of_text_words)
    #
    # # ****************************************************************************************************
    # # save each document distribution
    # # ****************************************************************************************************
    # name = '/document_topics_%s.p' % percentage
    # f1 = open(directory+name, 'w')
    # pickle.dump(model.doc_topic_, f1, 2)
    # f1.close()
    #
    # name = '/topic_words_%s.p' % percentage
    # f1 = open(directory+name, 'w')
    # pickle.dump(model.topic_word_, f1, 2)
    # f1.close()
    #
    # # ****************************************************************************************************
    # # investigate the relevant words in each topic, and see which documents are classified into each topic
    # # ****************************************************************************************************
    # true_labels, pred_labels, relevant_words[percentage] =  investigate_topics(model, loaded_data, probability_of_words, language_indices, _lambda, n_top_words=30)
    #
    # invesitgate_videos_given_topics(true_labels, pred_labels, ordered_list_of_video_names)
    # print "\nvideos classified:", len(true_labels), len(pred_labels)


    # ****************************************************************************************************
    # print results
    # ****************************************************************************************************
    n, l, v, h, c, mi, nmi, a, mat, labs = print_results(true_labels, pred_labels, n_topics)

    # c = sorted(zip(true_labels, pred_labels))
    # true_labels, pred_labels = zip(*c)
    # for i,j in zip(true_labels, pred_labels):
    #     print i,j

    # ****************************************************************************************************
    # do a final E-step on all the data :)
    # ****************************************************************************************************
    rhot = pow(olda._tau0 + olda._updatect, -olda._kappa)
    olda._rhot = rhot
    (gamma, sstats) = olda.do_e_step(wordids, wordcts)

    import pdb; pdb.set_trace()

    y_true, y_pred = [], []
    for n, gam in enumerate(gamma):
        gam = gam / float(sum(gam))

        if max(gam) > class_thresh:
            # thresholded_true.append(ordered_labels[n])
            y_true.append(ordered_labels[n])
            y_pred.append(np.argmax(gam))

    n, l, v, h, c, mi, nmi, a, mat, labs = print_results(y_true, y_pred, n_topics)

    # ****************************************************************************************************
    # Write out a file of results
    # ****************************************************************************************************
    name = '/results_vb.txt'
    f1 = open(directory+name, 'w')
    f1.write('n_topics: %s \n' % n_topics)
    f1.write('n_iters: %s \n' % n_iters)
    f1.write('dirichlet_params: (%s, %s) \n' % (dirichlet_params[0], dirichlet_params[1]))
    f1.write('class_thresh: %s \n' % class_thresh)
    f1.write('code book length: %s \n' % codebook_lengh)
    # f1.write('sum of all words: %s \n' % sum_of_all_words)
    f1.write('videos classified: %s \n \n' % len(pred_labels))

    f1.write('v-score: %s \n' % v)
    f1.write('homo-score: %s \n' % h)
    f1.write('comp-score: %s \n' % c)
    f1.write('mi: %s \n' % mi)
    f1.write('nmi: %s \n \n' % nmi)
    f1.write('mat: \n')

    headings = ['{:3d}'.format(int(r)) for r in xrange(n_topics)]
    f1.write('T = %s \n \n' % headings)
    for row, lab in zip(mat, labs):
        text_row = ['{:3d}'.format(int(r)) for r in row]
        f1.write('    %s : %s \n' % (text_row, lab))
    f1.write('\n')
    # f1.write('relevant_words: \n')
    # for i, words in relevant_words[percentage].items():
    #     f1.write('Topic %s : %s \n' % (i, words[:10]))
    f1.close()
