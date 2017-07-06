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

class language():
    """docstring for shapes."""
    def __init__(self):
        self.dir = "/home/omari/Datasets/Baxter_Dataset/scene"
        self.dir_save = "/home/omari/Datasets/Baxter_Dataset_final/features/language/"
        self.dir_cluster = "/home/omari/Datasets/Baxter_Dataset_final/features/"
        self.features = ["colours"]
        self.unique_words = []
        self.n_per_video = {}
        self.cluster_count = {}
        self.ngram_count = {}
        self.K = []
        self.GT_dict = {}
        self.f_score = []
        self.x_axis_data = []
        self.toggle = 1

    def _read_sentences(self):
        self.bad_ngrams = []
        for video in range(1,205):
            self.n_per_video[video] = []
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
                # print video,line
                words = line.split(" ")
                for word in words:
                    if word not in self.unique_words:
                        self.unique_words.append(word)
                        self.ngram_count[self.unique_words.index(word)] = 0
                    if word not in self.n_per_video[video]:
                        self.n_per_video[video].append(self.unique_words.index(word))
        # print self.n_per_video
        for video in range(1,205):
            for word in self.n_per_video[video]:
                self.ngram_count[word]+=1
        # print self.ngram_count
        self.bad_ngrams.append(self.unique_words.index("the"))
        keys = self.ngram_count.keys()
        for k in keys:
            if self.ngram_count[k]<3:
                self.bad_ngrams.append(k)
        self.total_n_num = len(self.unique_words)

    def _read_clusters(self):
        self.c_per_video = {}
        self.cluster_num = {}
        self.total_c_num = 0
        self.unique_clusters = []
        self.bad_clusters = []
        for f in self.features:
            self.cluster_num[f], self.c_per_video[f] = pickle.load(open(self.dir_cluster+f+"/clusters_per_video.p", "rb" ))
            self.total_c_num += self.cluster_num[f]
            for i in range(self.cluster_num[f]):
                self.unique_clusters.append(f+"_"+str(i))
                self.cluster_count[f+"_"+str(i)] = 0
        for f in self.features:
            for v in range(1,205):
                for i in self.c_per_video[f][v]:
                    self.cluster_count[f+"_"+str(i)]+=1
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
            self.GT_dict['colours_2'] =  ["brown"]
            self.GT_dict['colours_3'] =  []
            self.GT_dict['colours_4'] =  ["blue"]
            self.GT_dict['colours_5'] =  ["white"]
            self.GT_dict['colours_6'] =  ["green"]
            self.GT_dict['colours_7'] =  ["orange","red"]
            self.GT_dict['colours_8'] =  ["yellow"]
            self.GT_dict['colours_9'] =  ["red"]
            self.GT_dict['colours_10'] = ["yellow"]
            self.GT_dict['colours_11'] = ["green"]
            self.GT_dict['colours_12'] = ["blue"]
            self.GT_dict['colours_13'] = ["pink","purple"]
            self.GT_dict['colours_14'] = ["green"]
            self.GT_dict['colours_15'] = ["blue"]

        if "shapes" in self.features:
            self.GT_dict['shapes_0'] =  ["coffee"]
            self.GT_dict['shapes_1'] =  ["cup","block"]
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
            self.GT_dict['shapes_14'] = ["block"]
            self.GT_dict['shapes_15'] = ["mug"]
            self.GT_dict['shapes_16'] = ["octopus","dolphin","lemon"]
            self.GT_dict['shapes_17'] = ["cup","stapler"]
            self.GT_dict['shapes_18'] = ["bowl"]
            self.GT_dict['shapes_19'] = ["mug"]
            self.GT_dict['shapes_20'] = ["plate"]
            self.GT_dict['shapes_21'] = ["mug"]
            self.GT_dict['shapes_22'] = ["mug"]
            self.GT_dict['shapes_23'] = ["mug","coffee"]
            self.GT_dict['shapes_24'] = ["mug"]

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
                        self.K[row,col]+=1
                # print '-----------'
        for i in self.bad_clusters:
            row = self.unique_clusters.index(i)
            self.K[row,:] = 0
        for i in self.bad_ngrams:
            # print i,self.unique_words[i]
            self.K[:,i] = 0
        # print self.K[-1,:]
        # print self.unique_words.index("mug")

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
        x = self.x_axis_data
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

def main():
    L = language()
    L._read_sentences()
    L._read_clusters()
    L._get_GT()

    ## inital testing
    # L._build_K(1.0)
    # L._LP_assign(0.07,0)

    ## colour 0.035
    ## shape 0.07
    ## incremental analysis
    for data in range(1,6):
        L._build_K(2*data/10.0)
        L.x_axis_data.append(2*data/10.0)
        L._LP_assign(0.035,2)
    L._plot_f_score(0)

    # ## sensitivity analysis
    # for data in range(1,20):
    #     L._build_K(1.0)#(data)
    #     L.x_axis_data.append(data/100.0)
    #     L._LP_assign(data/100.0,1)
    # L._plot_f_score(1)

if __name__=="__main__":
    main()
