import numpy as np
from sklearn import mixture
import itertools
from sklearn.metrics.cluster import v_measure_score
import cv2
import pickle
import pulp
import sys
import colorsys
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import shuffle

class objects_class():
    """docstring for faces"""
    def __init__(self):
        # self.username = getpass.getuser()
        # self.dir1 = '/home/'+self.username+'/Datasets/ECAI_dataset/features/vid'
        self.dir2 = '/home/omari/Datasets/ECAI_dataset/objects/'
        self.dir_objects =  '/home/omari/Datasets/ECAI_dataset/features/vid'
        self.dir_grammar = '/home/omari/Datasets/ECAI_dataset/grammar/'
        self.dir_annotation = '/home/omari/Datasets/ECAI_dataset/ECAI_annotations/vid'
        self.im_len = 60
        self.f_score = []
        self.Pr = []
        self.Re = []

    def _read_objects(self):
        f = open(self.dir2+"per_video_objects.txt","r")
        self.objects = {}
        self.video_num = []
        self.all_objects = []
        for i in range(1,494):
            self.video_num.append(i)
            self.objects[i] = []

        count = 0
        count2 = 1
        for line in f:
            if count == 2:
                line1 = line.split('\n')[0]
            if count == 3:
                line2 = line.split('\n')[0]
            if count == 4:
                vals = map(float, line1.split(' '))
                obj_ids = map(int, line2.split(' '))
                for id in obj_ids:
                    if vals[id]>0:
                        if id not in self.all_objects:
                            self.all_objects.append(id)
                        self.objects[count2].append(id)
                count=0
                print self.objects[count2]
                count2+=1
                print '-----------'
            else:
                count+=1
        # print self.all_objects


    def _read_tags(self):
        self.objects_words = {}
        self.all_objects_words = []
        self.tags,self.words_count = pickle.load(open( self.dir_grammar+"tags_activity.p", "rb" ) )
        for i in self.tags.keys():
            self.objects_words[i] = []
            for word in self.tags[i]['object']:
                if word not in self.objects_words[i]:
                    self.objects_words[i].append(word)
                if str(word) not in self.all_objects_words:
                    self.all_objects_words.append(str(word))
        self.all_objects_words = sorted(self.all_objects_words)
        self.all_objects_words_copy = copy.copy(self.all_objects_words)

    def _get_groundTruth(self):
        self.GT_dict = {}
        self.GT_total_links = 0
        f = open(self.dir2+"/GroundTruth.txt",'r')
        for line in f:
            line = line.split("\n")[0]
            cluster,name = line.split(',')
            cluster = int(cluster)
            if cluster not in self.GT_dict:
                self.GT_dict[cluster] = []
            if name not in self.GT_dict[cluster]:
                self.GT_dict[cluster].append(name)
                self.GT_total_links+=1

    def _cluster_objects(self):
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
        self.Y_ = list(Y_)
        for i in range(len(self.final_clf.means_)):
            print i,self.Y_.count(i)

    def _read_objects_clusters(self):
        self.final_clf,X_ = pickle.load(open(self.dir2+'colour_clusters.p',"rb"))
        self.Y_ = self.final_clf.predict(self.X)
        # print self.Y_

    def _plot_objects_clusters(self):
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
            if i in [9]:
                self.rgb.append((1,0,0))
            else:
                self.rgb.append((0,0,0))

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

    def _assignment_matrix(self,fraction):
        self.all_objects_words = copy.copy(self.all_objects_words_copy)
        self.objects_words_count = {}
        self.cluster_count = {}
        stop = len(self.video_num)*fraction
        count = 0
        videos_seen = {}
        for vid in self.video_num:
            videos_seen[vid] = {}
            videos_seen[vid]['clusters'] = []
            videos_seen[vid]['object'] = []
        for vid in self.video_num:
            for cluster in self.objects[vid]:
                if cluster in videos_seen[vid]['clusters']:
                    continue
                videos_seen[vid]['clusters'].append(cluster)
                if cluster not in self.cluster_count:
                    self.cluster_count[cluster] = 0
                if count <= stop:
                    self.cluster_count[cluster] += 1
                for name in self.objects_words[vid]:
                    if name in self.all_objects_words:
                        noun_i = self.all_objects_words.index(name)
                        if noun_i not in self.objects_words_count:
                            self.objects_words_count[noun_i] = 0
                        if count <= stop:
                            if noun_i not in videos_seen[vid]['object']:
                                self.objects_words_count[noun_i]+=1
                                videos_seen[vid]['object'].append(noun_i)
            count += 1
        for i in self.objects_words_count:
            if not self.objects_words_count[i]:
                self.objects_words_count[i] = 1
        for i in self.cluster_count:
            if not self.cluster_count[i]:
                self.cluster_count[i] = 1
        # remove low counts
        objects_words_to_remove = []
        for i in self.objects_words_count:
            if self.objects_words_count[i]<3:
                objects_words_to_remove.append(i)
        for i in reversed(objects_words_to_remove):
            print '>>', self.all_objects_words[i]
            del self.all_objects_words[i]

        self.CM_nouns = np.zeros((len(self.all_objects_words),len(self.all_objects)))
        self.CM_clust = np.zeros((len(self.all_objects),len(self.all_objects_words)))
        self.objects_words_count = {}
        self.cluster_count = {}
        stop = len(self.video_num)*fraction
        count = 0
        videos_seen = {}
        for cluster,vid in zip(self.objects,self.video_num):
            videos_seen[vid] = {}
            videos_seen[vid]['clusters'] = []
            videos_seen[vid]['object'] = []

        for vid in self.video_num:
            for cluster_name in self.objects[vid]:
                cluster = self.all_objects.index(cluster_name)
                if cluster in videos_seen[vid]['clusters']:
                    continue
                videos_seen[vid]['clusters'].append(cluster)
                if cluster not in self.cluster_count:
                    self.cluster_count[cluster] = 0
                if count <= stop:
                    self.cluster_count[cluster] += 1
                for name in self.objects_words[vid]:
                    # if name == 'printing':
                    #     print '>>>>>>>>',vid
                    #     print self.objects[vid]
                    if name in self.all_objects_words:
                        noun_i = self.all_objects_words.index(name)
                        if noun_i not in self.objects_words_count:
                            self.objects_words_count[noun_i] = 0
                        if count <= stop:
                            self.CM_nouns[noun_i,cluster] += 1
                            self.CM_clust[cluster,noun_i] += 1
                            if noun_i not in videos_seen[vid]['object']:
                                self.objects_words_count[noun_i]+=1
                                videos_seen[vid]['object'].append(noun_i)
            count += 1
        for i in self.objects_words_count:
            if not self.objects_words_count[i]:
                self.objects_words_count[i] = 1
        for i in self.cluster_count:
            if not self.cluster_count[i]:
                self.cluster_count[i] = 1

        print '--------------------'
        # print self.CM_nouns[self.all_objects_words.index("printing")],self.objects_words_count[self.all_objects_words.index("printing")]
        # print self.CM_clust[1]
        # pickle.dump( [self.CM_nouns, self.CM_clust, self.objects_words_count, self.cluster_count, self.all_objects_words], open( self.dir2+'objects_correlation.p', "wb" ) )

    def _pretty_plot(self):
        self.cluster_images = {}
        print '-------------------------------------',len(self.Y_),len(self.rgb)
        for rgb,val in zip(self.rgb,self.Y_):
            if val not in self.cluster_images:
                self.cluster_images[val] = []
            rgb = [rgb[0]+rgb[1]+rgb[2],int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255)]
            # if rgb not in self.cluster_images[val]:
            self.cluster_images[val].append(rgb)

        for val in self.cluster_images:
            self.cluster_images[val] = sorted(self.cluster_images[val])
            if len(self.cluster_images[val])>30:
                selected = []
                count = 0
                for i in range(0,len(self.cluster_images[val]),len(self.cluster_images[val])/19):
                    if count < 30:
                        selected.append(self.cluster_images[val][i])
                self.cluster_images[val] = selected
        image_cluster_total = np.zeros((self.im_len*5*7,self.im_len*5*5,3),dtype=np.uint8)+255
        count3 = 0
        for count2,p in enumerate(self.cluster_images):
            maxi = len(self.cluster_images[p])
            image_avg = np.zeros((self.im_len,self.im_len,3),dtype=np.uint8)
            image_cluster = np.zeros((self.im_len*5,self.im_len*5,3),dtype=np.uint8)+255
            # print maxi
            for count,rgb in enumerate(self.cluster_images[p]):
                img = np.zeros((self.im_len,self.im_len,3),dtype=np.uint8)
                img[:,:,0]+=rgb[3]
                img[:,:,1]+=rgb[2]
                img[:,:,2]+=rgb[1]
                # img[0:2,:,:]=0
                # img[-2:,:,:]=0
                # img[:,0:2,:]=0
                # img[:,-2:,:]=0
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
                cv2.imwrite(self.dir2+'all_clusters.jpg',image_cluster_total)
        #
        #     if p in [2,3,4]:
        #         i1x = np.mod(count3,3)*self.im_len*5
        #         i2x = (np.mod(count3,3)+1)*self.im_len*5
        #         count3+=1
        #         i1y = 0
        #         i2y = self.im_len*5
        #         paper_img[i1y:i2y,i1x:i2x,:] = image_cluster
        #         cv2.imwrite(self.dir_faces+'faces_clusters_ex.jpg',paper_img)
        #
            cv2.imwrite(self.dir2+str(p)+'_cluster.jpg',image_cluster)
        #     cv2.imwrite(self.dir_faces+'cluster_images/'+str(p)+'_cluster.jpg',image_cluster)

    def _LP_assign(self,max_assignments):
        Precision = 0
        Recall = 0
        if max_assignments != 0:
            faces = self.cluster_count.keys()
            words = self.all_objects_words
            # print words
            def word_strength(face, word):
                # print face,word,words.index(word)
                A = 100.0*self.CM_nouns[words.index(word)][face]/self.cluster_count[face]
                if words.index(word) in self.objects_words_count:
                    B = 100.0*self.CM_nouns[words.index(word)][face]/self.objects_words_count[words.index(word)]
                else:
                    B=0
                #conditional probabiltiy: (N(w,f)/N(f) + N(w,f)/N(w)) /2
                # return round((100.0*self.CM_nouns[words.index(word)][face]/self.cluster_count[face] + 100.0*self.CM_nouns[words.index(word)][face]/self.objects_words_count[words.index(word)])/2)
                return round(max(A , B))

            possible_assignments = [(x,y) for x in faces for y in words]
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
            #each face should get at least one assignment
            for face in faces:
                prob += sum([x[assignment] for assignment in possible_assignments
                                            if face==assignment[0] ]) >= 1, "Must_assign_face_%d"%face
            prob.solve()
            # print ([sum([pulp.value(x[assignment]) for assignment in possible_assignments if face==assignment[0] ]) for face in faces])
            # f = open(self.dir_faces+"circos/faces.txt","w")
            correct = 0
            for assignment in possible_assignments:
                if x[assignment].value() == 1.0:
                    print self.all_objects[assignment[0]],assignment[1]
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
        # # print max_assignments
        # print self.f_score
        # # print '-----------'
        # pickle.dump( self.f_score, open( self.dir2+'objects_incremental.p', "wb" ) )
        # # pickle.dump( self.f_score, open( self.dir_faces+'faces_f_score3.p', "wb" ) )

    def _plot_incremental(self):
        self.f_score = pickle.load(open(self.dir2+'objects_incremental.p',"rb"))
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
            x.append(i/float(len(self.cluster_count.keys())*len(self.all_objects_words)))
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
    f = objects_class()
    f._read_objects()
    f._read_tags()
    # f._cluster_objects()
    # f._read_objects_clusters()
    # f._plot_objects_clusters()
    # f._get_groundTruth()
    f._assignment_matrix(1.0)
    f._LP_assign(.05)
    # f.max = 10
    # for i in range(1,f.max+1):
    #     print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',i/float(f.max)
    #     f._assignment_matrix(i/float(f.max))
    #     f._LP_assign(.05)
    # f._plot_incremental()
    # # f._plot_f_score()
    # f._pretty_plot()

if __name__=="__main__":
    main()
