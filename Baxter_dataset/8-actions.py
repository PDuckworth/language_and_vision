import cv2
import numpy as np
import glob
from operator import add
import pickle
from sklearn import mixture
# import itertools
from sklearn.metrics.cluster import v_measure_score
from sklearn import metrics
from sklearn import svm
import pulp
import matplotlib.pyplot as plt
from random import randint

class language():
    """docstring for shapes."""
    def __init__(self):
        self.dir_sensitivity = '/home/omari/Datasets/sensitivity/'
        self.dir = "/home/omari/Datasets/Baxter_Dataset/scene"
        self.dir_save = "/home/omari/Datasets/Baxter_Dataset_final/features/language/"
        self.dir_cluster = "/home/omari/Datasets/Baxter_Dataset_final/features/"
        self.dir_scale = "/home/omari/Datasets/scalibility/Baxter/"
        self.features = ["distances"]
        self.unique_words = []
        self.unique_words_incremental = []
        self.n_per_video = {}
        self.cluster_count = {}
        self.ngram_count = {}
        self.K = []
        self.GT_dict = {}
        self.f_score = []
        self.x_axis = []
        self.toggle = 1
        # self.action_words = ["pick", "lift","place","move","put","stack"]
        self.pick_words = ["pick", "lift"]
        self.move_words = ["place","move","stack"]
        self.put_words = ["put"]

    def _read_sentences(self):

        all_data = []
        all_GT = []
        data = []
        GT = []
        test = []
        GT_test = []
        for video in range(1,205):
            dir1 = self.dir+str(video)+"/annotation/natural_language_commands.txt"
            f = open(dir1,'r')
            for line in f:
                line = line.split("\n")[0]
                if "pick up" in line:
                    if self.toggle:
                        line = line.replace("pick up", "lift")
                        self.toggle = 0
                    else:
                        self.toggle = 1
                action = -1
                for word in self.pick_words:
                    if word in line:
                        d1 = randint(0, 2)
                        d = d1+ np.random.normal(0, .3, 1)
                        action = "pick"
                for word in self.move_words:
                    if word in line:
                        d1 = 3
                        d = 3+ np.random.normal(0, .3, 1)
                        action = "move"
                for word in self.put_words:
                    if word in line:
                        d1 = randint(4, 6)
                        d = d1+ np.random.normal(0, .3, 1)
                        action = "put"
                if all_data == []:
                    all_data = [d1]
                else:
                    all_data = np.vstack((all_data,d1))
                all_GT.append(action)
                if video < 160:
                    if data == []:
                        data = [d]
                    else:
                        data = np.vstack((data,d))
                    GT.append(action)
                else:
                    if test == []:
                        test = [d]
                    else:
                        test = np.vstack((test,d))
                    GT_test.append(action)

        self._svm(data, GT, test, GT_test)
        self._cluster_data(all_data, all_GT, "name", 10)

    def _svm(self,x,y,x_test,y_test):
        mean = 0
        for i in range(150):
            clf = svm.SVC(kernel='linear')
            clf.fit(x, y)
            A = clf.predict(x_test)
            mean += metrics.v_measure_score(y_test, A)
        mean/=150
        print("this varies significantly supervised V-measure: %0.2f" % mean)

    def _cluster_data(self, X, GT, name, n):
        best_v = 0
        lowest_bic = 10000000000
        for i in range(5):
            print '#####',i
            n_components_range = range(3, n)
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
        # pickle.dump( [final_Y_, best_gmm], open( '/home/omari/Datasets/Dukes_modified/results/'+name+'_clusters.p', "wb" ) )

        self._print_results(GT,final_Y_,best_gmm)

    def _print_results(self, GT,Y_,best_gmm):
        #print v_measure_score(GT, Y_)
        true_labels = GT
        pred_labels = Y_
        print "\n dataset unique labels:", len(set(true_labels))
        print "number of clusters:", len(best_gmm.means_)
        print("Mutual Information: %.2f" % metrics.mutual_info_score(true_labels, pred_labels))
        print("Adjusted Mutual Information: %0.2f" % metrics.normalized_mutual_info_score(true_labels, pred_labels))
        print("Homogeneity: %0.2f" % metrics.homogeneity_score(true_labels, pred_labels))
        print("Completeness: %0.2f" % metrics.completeness_score(true_labels, pred_labels))
        print("V-measure: %0.2f" % metrics.v_measure_score(true_labels, pred_labels))

    def _read_clusters(self):
        self.c_per_video = {}
        self.cluster_num = {}
        self.total_c_num = 0
        self.unique_clusters = []
        self.bad_clusters = []
        for f in self.features:
            self.cluster_num[f], self.c_per_video[f] = pickle.load(open(self.dir_cluster+f+"/clusters_per_video.p", "rb" ))
            # print self.c_per_video[f]
            self.total_c_num += self.cluster_num[f]
            for i in range(self.cluster_num[f]):
                self.unique_clusters.append(f+"_"+str(i))
                self.cluster_count[f+"_"+str(i)] = 0
        for f in self.features:
            for v in range(1,205):
                for i in self.c_per_video[f][v]:
                    self.cluster_count[f+"_"+str(i)]+=1
        print self.cluster_count
        keys = self.cluster_count.keys()
        for k in keys:
            if self.cluster_count[k]<3:
                self.bad_clusters.append(k)

    def _get_GT(self):
        self.GT_total_links = 0
        self.GT_dict = {}
        if "colours" in self.features:
            self.GT_dict['colours_0'] =  ["red"]
            self.GT_dict['colours_1'] =  []
            self.GT_dict['colours_2'] =  ["brown","black","coffee"]
            self.GT_dict['colours_3'] =  []
            self.GT_dict['colours_4'] =  ["blue","light"]
            self.GT_dict['colours_5'] =  ["white"]
            self.GT_dict['colours_6'] =  ["green"]
            self.GT_dict['colours_7'] =  ["orange","red"]
            self.GT_dict['colours_8'] =  ["yellow","lemon"]
            self.GT_dict['colours_9'] =  ["red"]
            self.GT_dict['colours_10'] = ["yellow"]
            self.GT_dict['colours_11'] = ["green"]
            self.GT_dict['colours_12'] = ["blue"]
            self.GT_dict['colours_13'] = ["pink","purple"]
            self.GT_dict['colours_14'] = ["green"]
            self.GT_dict['colours_15'] = ["blue"]

        if "shapes" in self.features:
            self.GT_dict['shapes_0'] =  ["coffee","cup"]
            self.GT_dict['shapes_1'] =  ["cup","cups"]
            self.GT_dict['shapes_2'] =  ["mug"]
            self.GT_dict['shapes_3'] =  ["block"]
            self.GT_dict['shapes_4'] =  ["block","bowl"]
            self.GT_dict['shapes_5'] =  ["mug"]
            self.GT_dict['shapes_6'] =  ["ball"]
            self.GT_dict['shapes_7'] =  ["block"]
            self.GT_dict['shapes_8'] =  ["block","mug"]
            self.GT_dict['shapes_9'] =  ["block","mug"]
            self.GT_dict['shapes_10'] = ["block","bowl"]
            self.GT_dict['shapes_11'] = ["block","mug"]
            self.GT_dict['shapes_12'] = ["block","mug"]
            self.GT_dict['shapes_13'] = ["bowl"]
            self.GT_dict['shapes_14'] = ["block","lemon"]
            self.GT_dict['shapes_15'] = ["mug"]
            self.GT_dict['shapes_16'] = ['apple','banana','carrot','dolphin','whale','duck','bird','octopus','egg','lemon']
            self.GT_dict['shapes_17'] = ["cup","stapler"]
            self.GT_dict['shapes_18'] = ["bowl","plate"]
            self.GT_dict['shapes_19'] = ["mug","cup"]
            self.GT_dict['shapes_20'] = ["plate"]
            self.GT_dict['shapes_21'] = ["mug"]
            self.GT_dict['shapes_22'] = ["mug"]
            self.GT_dict['shapes_23'] = ["mug","coffee"]
            self.GT_dict['shapes_24'] = ["mug"]

        if "distances" in self.features:
            self.GT_dict['distances_0'] =  ["furthest","far"]
            self.GT_dict['distances_1'] =  ["top","of"]
            self.GT_dict['distances_2'] =  ["inside"]
            self.GT_dict['distances_3'] =  []

        for i in self.GT_dict:
            for j in self.GT_dict[i]:
                self.GT_total_links+=1

    def _build_K(self,data):
        self.K = np.zeros((self.total_c_num,self.total_n_num))
        for f in self.features:
            for v in range(1,205):
                if v > data*204: continue
                for r in self.c_per_video[f][v]:
                    cluster = f+"_"+str(r)
                    row = self.unique_clusters.index(cluster)
                    for col in self.n_per_video[v]:
                        # print v,row,col
                        # self.K[:,col]-=1
                        # self.K[row+1:,col]-=1
                        self.K[row,col]+=1
                # print '-----------'
        # for i in range(self.total_n_num):
        #     self.K[:,i] += np.abs(np.min(self.K[:,i]))
            # print self.K[:,i]
        for i in self.bad_clusters:
            row = self.unique_clusters.index(i)
            self.K[row,:] = 0
        for i in self.bad_ngrams:
            # print i,self.unique_words[i]
            self.K[:,i] = 0
        # print self.unique_words[0]
        # print self.K[:,0]
        # A = self.unique_words.index("front")
        # print self.K[:,A]
        # print self.ngram_count[self.unique_words.index("front")]

    def _LP_assign(self,max_assignments,option):
        Precision = 0
        Recall = 0
        if max_assignments != 0:
            clusters = self.unique_clusters
            words = self.unique_words

            def word_strength(cluster, word):
                cluster_i = self.unique_clusters.index(cluster)
                word_i = self.unique_words.index(word)
                #conditional probabiltiy: (N(w,f)/N(f) + N(w,f)/N(w)) /2
                # return round((100.0*self.K[words.index(word)][cluster]/self.cluster_count[cluster] + 100.0*self.K[words.index(word)][cluster]/self.ngram_count[words.index(word)])/2)
                A = 100*self.K[cluster_i][word_i]
                return round(max(A/self.cluster_count[cluster] , A/self.ngram_count[word_i]))

            possible_assignments = [(x,y) for x in clusters for y in words]
            max_assignments = int(len(possible_assignments)*max_assignments)
            #create a binary variable for assignments
            x = pulp.LpVariable.dicts('x', possible_assignments,
                                        lowBound = 0,
                                        upBound = 1,
                                        cat = pulp.LpInteger)
            prob = pulp.LpProblem("Assignment problem", pulp.LpMaximize)
            #main objective function
            prob += sum([word_strength(*assignment) * x[assignment] for assignment in possible_assignments])
            #limiting the number of assignments
            prob += sum([x[assignment] for assignment in possible_assignments]) <= max_assignments, \
                                        "Maximum_number_of_assignments"
            #each cluster should get at least one assignment
            for cluster in clusters:
                cluster_i = self.unique_clusters.index(cluster)
                prob += sum([x[assignment] for assignment in possible_assignments
                                            if cluster==assignment[0] ]) >= 1, "Must_assign_cluster_%d"%cluster_i
            prob.solve()
            # print ([sum([pulp.value(x[assignment]) for assignment in possible_assignments if cluster==assignment[0] ]) for cluster in clusters])
            # f = open(self.dir_clusters+"circos/clusters.txt","w")
            correct = 0
            self.assignments_to_save = {}
            for assignment in possible_assignments:
                if x[assignment].value() == 1.0:
                    print assignment
            if option:
                for assignment in possible_assignments:
                    if x[assignment].value() == 1.0:
                        if assignment[0] not in self.assignments_to_save:
                            self.assignments_to_save[assignment[0]] = []
                        self.assignments_to_save[assignment[0]].append(assignment[1])
                        # print assignment[0]
                        if assignment[1] in self.GT_dict[assignment[0]]:
                            correct += 1
                Precision = correct/float(max_assignments)
                Recall = correct/float(self.GT_total_links)
                # print Precision,Recall
                if not Precision and not Recall:
                    f_score=0
                else:
                    f_score = 2*(Precision*Recall)/(Precision+Recall)
            else:
                f_score = 0
            self.f_score.append(f_score)
            # self.Pr.append(Precision)
            # self.Re.append(Recall)
            # print max_assignments
            print self.f_score
            print '-----------'
            if option == 1:
                pickle.dump( self.f_score, open( self.dir_save+'_'.join(self.features)+'_sensitivity.p', "wb" ) )
            if option == 2:
                pickle.dump( self.f_score, open( self.dir_save+'_'.join(self.features)+'_incremental.p', "wb" ) )
            # for cluster,vid in zip(self.Y_,self.video_num):
            #     if vid == 106:
            #         print '>>>>>>>',cluster
            # pickle.dump( self.assignments_to_save, open( self.dir2+'colours_assignments.p', "wb" ) )
            # # pickle.dump( self.f_score, open( self.dir_clusters+'clusters_f_score3.p', "wb" ) )

    def _plot_f_score(self,sensitivity):
        if sensitivity:
            self.f_score = pickle.load(open(self.dir_save+'_'.join(self.features)+'_sensitivity.p',"rb"))
        else:
            self.f_score = pickle.load(open(self.dir_save+'_'.join(self.features)+'_incremental.p',"rb"))
        x = self.x_axis
        maxi = 0
        for i in range(len(self.f_score)):
            # x.append(i/float(self.total_c_num*self.total_n_num))
            if self.f_score[i]>maxi:
                maxi = self.f_score[i]
                print i,maxi
        y = self.f_score
        x.append(x[-1])
        y.append(0)
        fig, ax = plt.subplots()
        ax.fill(x, y, zorder=10)
        ax.grid(True, zorder=5)
        plt.show()

    def _plot_sensitivity(self):
        x = self.x_axis
        y = self.f_score
        fig, ax = plt.subplots()
        ax.plot(x, y, zorder=10)
        ax.grid(True, zorder=5)
        pickle.dump( [x,y], open( self.dir_sensitivity+'Baxter_'+'_'.join(self.features)+'_sensitivity.p', "wb" ) )
        plt.show()

def main():
    L = language()
    L._read_sentences()
    # L._read_clusters()
    # L._get_GT()

    # inital testing
    # L._build_K(1.0)
    # L._LP_assign(0.07,0)

    ## incremental analysis
    # L.values = {}
    # L.values["colours"] = 0.07
    # L.values["shapes"] = 0.04
    # L.values["distances"] = 0.07
    # for data in range(1,6):
    #     L._build_K(2*data/10.0)
    #     L.x_axis.append(2*data/10.0)
    #     L._LP_assign(L.values[L.features[0]],2)
    # L._plot_f_score(0)

    # # ## sensitivity analysis
    # for data in range(1,30):
    #     L._build_K(1.0)#(data)
    #     L.x_axis.append(data/100.0)
    #     L._LP_assign(data/100.0,1)
    # L._plot_sensitivity()
    # L._plot_f_score(1)

if __name__=="__main__":
    main()
