import numpy as np
from sklearn import mixture
import itertools
from sklearn.metrics.cluster import v_measure_score
import cv2
import pickle
import pulp
import sys

import numpy as np
import matplotlib.pyplot as plt

class faces_class():
    """docstring for faces"""
    def __init__(self):
        self.dir_faces =  '/home/omari/Datasets/ECAI_dataset/faces/'
        self.dir_grammar = '/home/omari/Datasets/ECAI_dataset/grammar/'
        self.dir_annotation = '/home/omari/Datasets/ECAI_dataset/ECAI_annotations/vid'
        self.im_len = 60
        self.f_score = []
        self.Pr = []
        self.Re = []

    def _read_faces(self):
        f = open(self.dir_faces+'faces3_projections.csv','rb')
        self.faces = {}
        self.X = []
        self.persons = []

        for line in f:
            line = line.split('\n')[0]
            GT = line.split(',')[0]
            self.persons.append(GT)
            data = line.split(',')[1:]
            if GT not in self.faces:
                self.faces[GT] = []
            self.faces[GT].append(data)
            if self.X == []:
                self.X.append(data)
            else:
                self.X = np.vstack((self.X,data))

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
        self.words_top = {}
        self.words_low = {}
        self.nouns = {}
        self.all_nouns = []
        self.tags,self.words_count = pickle.load(open( self.dir_grammar+"tags.p", "rb" ) )
        for i in self.tags.keys():
            self.words_top[i] = []
            self.words_low[i] = []
            for word in self.tags[i]['upper_garment']:
                if word not in self.words_top[i]:
                    self.words_top[i].append(word)
            for word in self.tags[i]['lower_garment']:
                if word not in self.words_low[i]:
                    self.words_low[i].append(word)
            if i in self.video_num:
                self.nouns[i] = []
                for word in self.tags[i]['name']:
                    if str(word) not in self.nouns[i]:
                        self.nouns[i].append(str(word))
                    if str(word) not in self.all_nouns:
                        self.all_nouns.append(str(word))
        self.all_nouns = sorted(self.all_nouns)

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

    def _cluster_faces(self):
        final_clf = 0
        best_v = 0

        for i in range(25):
            print '#####',i
            n_components_range = range(30, 35)
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
            v_meas = v_measure_score(self.persons, Y_)
            if v_meas > best_v:
                best_v = v_meas
                final_clf = clf
                print best_v
        self.best_v = best_v
        Y_ = final_clf.predict(self.X)
        self.predictions = {}
        for person in self.faces:
            self.predictions[person] = np.zeros(len(final_clf.means_))
            for i in self.faces[person]:
                self.predictions[person][final_clf.predict([i])[0]]+=1

        self.cost_matrix = np.zeros((len(self.faces),len(final_clf.means_)))

        for count,person in enumerate(self.predictions):
            self.predictions[person]/=np.sum(self.predictions[person])
            self.cost_matrix[count] = self.predictions[person]

        self.cluster_images = {}
        for img,val in zip(self.images,Y_):
            if val not in self.cluster_images:
                self.cluster_images[val] = []
            self.cluster_images[val].append(img)

        rows = 0
        maxi = 0
        for p in self.cluster_images:
            if len(self.cluster_images[p])>maxi:
                maxi = len(self.cluster_images[p])

        maxi = int(np.sqrt(maxi))
        for p in self.cluster_images:
            count = 0
            image = np.zeros((self.im_len*maxi,self.im_len*maxi,3),dtype=np.uint8)+255
            image_avg = np.zeros((self.im_len,self.im_len,3),dtype=np.uint8)
            for i in range(maxi):
                for j in range(maxi):
                    if count < len(self.cluster_images[p]):
                        image[i*self.im_len:(i+1)*self.im_len,j*self.im_len:(j+1)*self.im_len,:] = self.cluster_images[p][count]
                        image_avg += self.cluster_images[p][count]/(len(self.cluster_images[p])+1)
                    image[i*self.im_len:i*self.im_len+1,  j*self.im_len:(j+1)*self.im_len,  :] = 0
                    image[(i+1)*self.im_len:(i+1)*self.im_len+1,  j*self.im_len:(j+1)*self.im_len,  :] = 0
                    image[i*self.im_len:(i+1)*self.im_len,  j*self.im_len:j*self.im_len+1,  :] = 0
                    image[i*self.im_len:(i+1)*self.im_len,  (j+1)*self.im_len:(j+1)*self.im_len+1,  :] = 0
                    count+=1
            # cv2.imshow('test',image)
            # cv2.waitKey(2000)
            cv2.imwrite(self.dir_faces+str(p)+'.jpg',image)
            cv2.imwrite(self.dir_faces+str(p)+'_avg.jpg',image_avg)
        pickle.dump( [self.faces,final_clf,self.X,self.best_v], open( self.dir_faces+'faces_clusters.p', "wb" ) )

    def _read_faces_clusters(self):
        self.faces,self.final_clf,self.X,self.best_v = pickle.load(open(self.dir_faces+'faces_clusters.p',"rb"))
        self.Y_ = self.final_clf.predict(self.X)

    def _assignment_matrix(self,fraction):
        self.CM_nouns = np.zeros((len(self.all_nouns),self.final_clf.n_components))
        self.CM_clust = np.zeros((self.final_clf.n_components,len(self.all_nouns)))

        self.nouns_count = {}
        self.cluster_count = {}
        stop = len(self.video_num)*fraction
        count = 0
        for cluster,vid in zip(self.Y_,self.video_num):
            if cluster not in self.cluster_count:
                self.cluster_count[cluster] = 0
            if count <= stop:
                self.cluster_count[cluster] += 1
            for name in self.nouns[vid]:
                noun_i = self.all_nouns.index(name)
                if noun_i not in self.nouns_count:
                    self.nouns_count[noun_i] = 0
                if count <= stop:
                    self.CM_nouns[noun_i,cluster] += 1
                    self.CM_clust[cluster,noun_i] += 1
                    self.nouns_count[noun_i]+=1
            count += 1
        for i in self.nouns_count:
            if not self.nouns_count[i]:
                self.nouns_count[i] = 1
        for i in self.cluster_count:
            if not self.cluster_count[i]:
                self.cluster_count[i] = 1
        print '--------------------'
        pickle.dump( [self.CM_nouns, self.CM_clust, self.nouns_count, self.cluster_count, self.all_nouns], open( self.dir_faces+'faces_correlation.p', "wb" ) )

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
            faces = range(33)
            words=self.all_nouns
            def word_strength(face, word):
                #conditional probabiltiy: (N(w,f)/N(f) + N(w,f)/N(w)) /2
                # return round((100.0*self.CM_nouns[words.index(word)][face]/self.cluster_count[face] + 100.0*self.CM_nouns[words.index(word)][face]/self.nouns_count[words.index(word)])/2)
                return round(max(100.0*self.CM_nouns[words.index(word)][face]/self.cluster_count[face] , 100.0*self.CM_nouns[words.index(word)][face]/self.nouns_count[words.index(word)]))
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
            # for face in faces:
            #     prob += sum([x[assignment] for assignment in possible_assignments
            #                                 if face==assignment[0] ]) >= 1, "Must_assign_face_%d"%face
            prob.solve()
            # print ([sum([pulp.value(x[assignment]) for assignment in possible_assignments if face==assignment[0] ]) for face in faces])
            f = open(self.dir_faces+"circos/faces.txt","w")
            correct = 0
            for assignment in possible_assignments:
                if x[assignment].value() == 1.0:
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
        self.Pr.append(Precision)
        self.Re.append(Recall)
        print max_assignments
        print self.f_score
        # print '-----------'
        pickle.dump( self.f_score, open( self.dir_faces+'faces_incremental.p', "wb" ) )
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
            x.append(i/float(len(self.cluster_count.keys())*len(self.all_nouns)))
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
    f = faces_class()
    f._read_faces()
    f._read_faces_images()
    f._read_tags()
    # # f._cluster_faces()
    f._read_faces_clusters()
    f._get_groundTruth()
    f.max = 10
    for i in range(1,f.max+1):
        f._assignment_matrix(i/float(f.max))
        f._LP_assign(.05)
    f._plot_incremental()
    # f.min = 0
    # f.max = len(f.cluster_count.keys())*len(f.all_nouns)
    # for i in range(f.min,f.max):
    # f._plot_f_score()
    # f._pretty_plot()

if __name__=="__main__":
    main()
