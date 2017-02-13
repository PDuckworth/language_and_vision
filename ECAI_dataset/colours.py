import numpy as np
from sklearn import mixture
import itertools
from sklearn.metrics.cluster import v_measure_score
import cv2
import pickle
import pulp
import sys
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class colours_class():
    """docstring for faces"""
    def __init__(self):
        # self.username = getpass.getuser()
        # self.dir1 = '/home/'+self.username+'/Datasets/ECAI_dataset/features/vid'
        self.dir2 = '/home/omari/Datasets/ECAI_dataset/colours/'
        self.dir_colours =  '/home/omari/Datasets/ECAI_dataset/features/vid'
        self.dir_grammar = '/home/omari/Datasets/ECAI_dataset/grammar/'
        self.dir_annotation = '/home/omari/Datasets/ECAI_dataset/ECAI_annotations/vid'
        self.im_len = 60
        self.f_score = []
        self.Pr = []
        self.Re = []

    def _read_colours(self):
        self.X = []
        self.rgb = []
        self.video_num = []
        for f in range(139,140):
            count1 = 0
            count2 = 0
            top = 1
            f = open(self.dir_colours+str(f)+'/colours.txt','r')
            for line in f:
                line = line.split('\n')[0]
                if '-' in line:
                    top = 0
                else:
                    data = []
                    line = line.split(',')
                    if top and count1<3:
                        count1+=1
                        data = map(int, line)

                    if not top and count2<3:
                        count2+=1
                        data = map(int, line)

                    if data != []:
                        hls = colorsys.rgb_to_hls(data[2]/255.0, data[1]/255.0, data[0]/255.0)
                        xyz = self._hls_to_xyz(hls)
                        self.video_num.append(f)
                        if self.X == []:
                            self.X = [xyz]
                            self.rgb = [(data[2]/255.0, data[1]/255.0, data[0]/255.0)]
                        else:
                            self.X = np.vstack((self.X,xyz))
                            self.rgb.append((data[2]/255.0, data[1]/255.0, data[0]/255.0))
                        # print self.X

    def _hls_to_xyz(self,hls):
        h = hls[0]*2*np.pi
        l = hls[1]
        s = hls[2]
        z = l
        x = s*np.cos(h)
        y = s*np.sin(h)
        return [x,y,z]

    def _read_faces_images(self):
        f = open(self.dir_faces+'faces_images.csv','rb')
        self.video_num = []
        self.images = []
        for line in f:
            line = line.split('\n')[0]
            vid = line.split('/')[1].split('_')[1]
            if vid not in self.video_num:
                self.video_num.append(int(vid))
            img = cv2.imread(self.dir_faces+line)
            self.images.append(cv2.resize(img, (self.im_len,self.im_len), interpolation = cv2.INTER_AREA))
            # cv2.imshow('test',img)
            # cv2.waitKey(100)
        # print self.video_num

    def _read_tags(self):
        # self.words_top = {}
        # self.words_low = {}
        self.adj = {}
        self.all_adj = []
        self.tags,self.words_count = pickle.load(open( self.dir_grammar+"tags.p", "rb" ) )
        for i in self.tags.keys():
            # self.words_top[i] = []
            # self.words_low[i] = []
            self.adj[i] = []
            for word in self.tags[i]['upper_garment']:
                if word not in self.adj[i]:
                    self.adj[i].append(word)
                if str(word) not in self.all_adj:
                    self.all_adj.append(str(word))
            for word in self.tags[i]['lower_garment']:
                if word not in self.adj[i]:
                    self.adj[i].append(word)
                if str(word) not in self.all_adj:
                    self.all_adj.append(str(word))
        self.all_adj = sorted(self.all_adj)
        print self.all_adj

    def _get_groundTruth(self):
        self.GT_dict = {}
        self.GT_total_links = 0
        for cluster,vid in zip(self.Y_,self.video_num):
            if cluster not in self.GT_dict:
                self.GT_dict[cluster] = []
            f = open(self.dir_annotation+str(vid)+"/person.txt",'r')
            name = f.readline().split('\n')[0].split('#')[1]
            name = str(name.lower())
            # print vid,cluster,name
            if name not in self.GT_dict[cluster]:
                self.GT_dict[cluster].append(name)
                self.GT_total_links+=1
        # print self.GT_dict

    def _cluster_colours(self):
        final_clf = 0
        best_v = 0

        n_components_range = range(5, 15)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        lowest_bic = np.infty
        for cv_type in cv_types:
            for n_components in n_components_range:
                gmm = mixture.GaussianMixture(n_components=n_components,covariance_type=cv_type)
                gmm.fit(self.X)
                bic = gmm.bic(self.X)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
        clf = best_gmm
        Y_ = clf.predict(self.X)
        pickle.dump( [clf,self.X], open( self.dir2+'colour_clusters.p', "wb" ) )
        self.final_clf = clf
        self.Y_ = Y_

    def _read_colours_clusters(self):
        self.final_clf,X_ = pickle.load(open(self.dir2+'colour_clusters.p',"rb"))
        self.Y_ = self.final_clf.predict(self.X)
        # print self.Y_

    def _plot_colours_clusters(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = []
        Y = []
        Z = []
        for x,y,z in self.X:
            X.append(x)
            Y.append(y)
            Z.append(z)
        ax.scatter(X,Y,Z, c=self.rgb, marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        self._HSV_tuples = [(x*1.0/11, 1.0, .7) for x in range(len(self.final_clf.means_))]
        self._colors = map(lambda x: colorsys.hsv_to_rgb(*x), self._HSV_tuples)

        self.rgb = []
        for i in self.Y_:
            # if i in [8,10,12]:
                self.rgb.append(self._colors[i])
            # else:
            #     self.rgb.append((0,0,0))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = []
        Y = []
        Z = []
        for x,y,z in self.X:
            X.append(x)
            Y.append(y)
            Z.append(z)
        ax.scatter(X,Y,Z, c=self.rgb, marker='o')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()
        print self.Y_

    def _assignment_matrix(self,fraction):

        # self.adj_count = {}
        # self.cluster_count = {}
        # stop = len(self.video_num)*fraction
        # count = 0
        # for cluster,vid in zip(self.Y_,self.video_num):
        #     if cluster not in self.cluster_count:
        #         self.cluster_count[cluster] = 0
        #     if count <= stop:
        #         self.cluster_count[cluster] += 1
        #     for name in self.adj[vid]:
        #         noun_i = self.all_adj.index(name)
        #         if noun_i not in self.adj_count:
        #             self.adj_count[noun_i] = 0
        #         if count <= stop:
        #             # self.CM_nouns[noun_i,cluster] += 1
        #             # self.CM_clust[cluster,noun_i] += 1
        #             self.adj_count[noun_i]+=1
        #     count += 1
        # for i in self.adj_count:
        #     if not self.adj_count[i]:
        #         self.adj_count[i] = 1
        # for i in self.cluster_count:
        #     if not self.cluster_count[i]:
        #         self.cluster_count[i] = 1
        #
        # # remove low counts
        # adj_to_remove = []
        # for i in self.adj_count:
        #     if self.adj_count[i]<3:
        #         adj_to_remove.append(i)
        # for i in reversed(adj_to_remove):
        #     # print '>>>>>>>', self.all_adj[i]
        #     del self.all_adj[i]

        self.CM_nouns = np.zeros((len(self.all_adj),self.final_clf.n_components))
        self.CM_clust = np.zeros((self.final_clf.n_components,len(self.all_adj)))
        self.adj_count = {}
        self.cluster_count = {}
        stop = len(self.video_num)*fraction
        count = 0
        for cluster,vid in zip(self.Y_,self.video_num):
            if cluster not in self.cluster_count:
                self.cluster_count[cluster] = 0
            if count <= stop:
                self.cluster_count[cluster] += 1
            for name in self.adj[vid]:
                if name == "red":
                    print '>>>>>>>>>>>>',vid
                if name in self.all_adj:
                    noun_i = self.all_adj.index(name)
                    if noun_i not in self.adj_count:
                        self.adj_count[noun_i] = 0
                    if count <= stop:
                        self.CM_nouns[noun_i,cluster] += 1
                        self.CM_clust[cluster,noun_i] += 1
                        self.adj_count[noun_i]+=1
            count += 1
        for i in self.adj_count:
            if not self.adj_count[i]:
                self.adj_count[i] = 1
        for i in self.cluster_count:
            if not self.cluster_count[i]:
                self.cluster_count[i] = 1

        print '--------------------'
        print self.CM_nouns[self.all_adj.index("red")],self.adj_count[self.all_adj.index("red")]
        pickle.dump( [self.CM_nouns, self.CM_clust, self.adj_count, self.cluster_count, self.all_adj], open( self.dir2+'colours_correlation.p', "wb" ) )

    def _pretty_plot(self):
        self.cluster_images = {}
        for img,val in zip(self.images,self.Y_):
            if val not in self.cluster_images:
                self.cluster_images[val] = []
            self.cluster_images[val].append(img)

        image_cluster_total = np.zeros((self.im_len*5*7,self.im_len*5*5,3),dtype=np.uint8)+255
        paper_img = np.zeros((self.im_len*5,self.im_len*5*3,3),dtype=np.uint8)+255
        # print len(self.cluster_images)
        # print iii
        count3 = 0
        for count2,p in enumerate(self.cluster_images):
            maxi = len(self.cluster_images[p])
            image_avg = np.zeros((self.im_len,self.im_len,3),dtype=np.uint8)
            image_cluster = np.zeros((self.im_len*5,self.im_len*5,3),dtype=np.uint8)+255
            # print maxi
            for count,img in enumerate(self.cluster_images[p]):
                img[0:2,:,:]=0
                img[-2:,:,:]=0
                img[:,0:2,:]=0
                img[:,-2:,:]=0
                image_avg += img/(len(self.cluster_images[p])+1)
                ang = count/float(maxi)*2*np.pi
                xc = int(1.95*self.im_len*np.cos(ang))
                yc = int(1.95*self.im_len*np.sin(ang))
                # print xc,yc
                C = int(2.5*self.im_len)
                x1 = int(xc-self.im_len/2.0+2.5*self.im_len)
                x2 = x1+self.im_len
                y1 = int(yc-self.im_len/2.0+2.5*self.im_len)
                y2 = y1+self.im_len
                cv2.line(image_cluster,(int(y1+y2)/2,int(x1+x2)/2),(C,C),(20,20,20),2)
                # print x1,x2,y1,y2
                image_cluster[x1:x2,y1:y2,:] = img
            image_avg = cv2.resize(image_avg, (int(self.im_len*1.4),int(self.im_len*1.4)), interpolation = cv2.INTER_AREA)
            x1 = int((2.5-.7)*self.im_len)
            x2 = int(x1+1.4*self.im_len)
            image_cluster[x1:x2,x1:x2,:] = image_avg
            if count2<35:
                i1x = np.mod(count2,7)*self.im_len*5
                i2x = (np.mod(count2,7)+1)*self.im_len*5
                i1y = int(count2/7)*self.im_len*5
                i2y = (int(count2/7)+1)*self.im_len*5
                image_cluster_total[i1x:i2x,i1y:i2y,:] = image_cluster
                cv2.imwrite(self.dir_faces+'all_clusters.jpg',image_cluster_total)

            if p in [2,3,4]:
                i1x = np.mod(count3,3)*self.im_len*5
                i2x = (np.mod(count3,3)+1)*self.im_len*5
                count3+=1
                i1y = 0
                i2y = self.im_len*5
                paper_img[i1y:i2y,i1x:i2x,:] = image_cluster
                cv2.imwrite(self.dir_faces+'faces_clusters_ex.jpg',paper_img)

            cv2.imwrite(self.dir_faces+str(p)+'_cluster.jpg',image_cluster)
            cv2.imwrite(self.dir_faces+'cluster_images/'+str(p)+'_cluster.jpg',image_cluster)

    def _LP_assign(self,max_assignments):
        Precision = 0
        Recall = 0
        if max_assignments != 0:
            faces = self.cluster_count.keys()
            words = self.all_adj
            def word_strength(face, word):
                #conditional probabiltiy: (N(w,f)/N(f) + N(w,f)/N(w)) /2
                # return round((100.0*self.CM_nouns[words.index(word)][face]/self.cluster_count[face] + 100.0*self.CM_nouns[words.index(word)][face]/self.adj_count[words.index(word)])/2)
                return round(max(100.0*self.CM_nouns[words.index(word)][face]/self.cluster_count[face] , 100.0*self.CM_nouns[words.index(word)][face]/self.adj_count[words.index(word)]))
            possible_assignments = [(x,y) for x in faces for y in words]
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
            #each face should get at least one assignment
            # for face in faces:
            #     prob += sum([x[assignment] for assignment in possible_assignments
            #                                 if face==assignment[0] ]) >= 1, "Must_assign_face_%d"%face
            prob.solve()
            # print ([sum([pulp.value(x[assignment]) for assignment in possible_assignments if face==assignment[0] ]) for face in faces])
            # f = open(self.dir_faces+"circos/faces.txt","w")
            correct = 0
            for assignment in possible_assignments:
                if x[assignment].value() == 1.0:
                    print assignment
        #             if assignment[1] in self.GT_dict[assignment[0]]:
        #                 correct += 1
        #     Precision = correct/float(max_assignments)
        #     Recall = correct/float(self.GT_total_links)
        #     # print Precision,Recall
        #     if not Precision and not Recall:
        #         f_score=0
        #     else:
        #         f_score = 2*(Precision*Recall)/(Precision+Recall)
        # else:
        #     f_score = 0
        # self.f_score.append(f_score)
        # self.Pr.append(Precision)
        # self.Re.append(Recall)
        # print max_assignments
        # print self.f_score
        # # print '-----------'
        # pickle.dump( self.f_score, open( self.dir_faces+'faces_incremental.p', "wb" ) )
        # pickle.dump( self.f_score, open( self.dir_faces+'faces_f_score3.p', "wb" ) )

    def _plot_incremental(self):
        self.f_score = pickle.load(open(self.dir_faces+'faces_incremental.p',"rb"))
        x = np.arange(len(self.f_score))/float(self.max-1)*493
        fig, ax = plt.subplots()
        ax.plot(x, self.f_score,'b',linewidth=2)
        # ax.plot(x, yp,'r')
        # ax.plot(x, yr,'g')
        ax.grid(True, zorder=5)
        plt.show()

    def _plot_f_score(self):
        self.f_score = pickle.load(open(self.dir_faces+'faces_f_score.p',"rb"))
        x = []
        maxi = 0
        for i in range(len(self.f_score)):
            x.append(i/float(len(self.cluster_count.keys())*len(self.all_adj)))
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
    f = colours_class()
    f._read_colours()
    f._read_tags()
    # f._cluster_colours()
    f._read_colours_clusters()
    f._plot_colours_clusters()
    f._assignment_matrix(1.0)
    # f._get_groundTruth()
    f._LP_assign(20)
    # f._plot_incremental()
    # f._plot_f_score()
    # f._pretty_plot()

if __name__=="__main__":
    main()
