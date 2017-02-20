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
        self.ok_clusters = []
        self.ok_videos = []

    def _get_video_per_days(self):
        self.video_per_day = {}
        self.video_per_day['2016-04-05'] = [('2016-04-05', 'vid437'), ('2016-04-05', 'vid400'), ('2016-04-05', 'vid465'), ('2016-04-05', 'vid486'), ('2016-04-05', 'vid376'), ('2016-04-05', 'vid441'), ('2016-04-05', 'vid458'), ('2016-04-05', 'vid374'), ('2016-04-05', 'vid481'), ('2016-04-05', 'vid489'), ('2016-04-05', 'vid365'), ('2016-04-05', 'vid429'), ('2016-04-05', 'vid451'), ('2016-04-05', 'vid454'), ('2016-04-05', 'vid436'), ('2016-04-05', 'vid361'), ('2016-04-05', 'vid351'), ('2016-04-05', 'vid372'), ('2016-04-05', 'vid446'), ('2016-04-05', 'vid388'), ('2016-04-05', 'vid392'), ('2016-04-05', 'vid466'), ('2016-04-05', 'vid409'), ('2016-04-05', 'vid460'), ('2016-04-05', 'vid358'), ('2016-04-05', 'vid453'), ('2016-04-05', 'vid359'), ('2016-04-05', 'vid382'), ('2016-04-05', 'vid490'), ('2016-04-05', 'vid364'), ('2016-04-05', 'vid395'), ('2016-04-05', 'vid352'), ('2016-04-05', 'vid350'), ('2016-04-05', 'vid484'), ('2016-04-05', 'vid394'), ('2016-04-05', 'vid348'), ('2016-04-05', 'vid399'), ('2016-04-05', 'vid405'), ('2016-04-05', 'vid450'), ('2016-04-05', 'vid448'), ('2016-04-05', 'vid416'), ('2016-04-05', 'vid487'), ('2016-04-05', 'vid411'), ('2016-04-05', 'vid387'), ('2016-04-05', 'vid360'), ('2016-04-05', 'vid421'), ('2016-04-05', 'vid383'), ('2016-04-05', 'vid424'), ('2016-04-05', 'vid377'), ('2016-04-05', 'vid455'), ('2016-04-05', 'vid483'), ('2016-04-05', 'vid369'), ('2016-04-05', 'vid438'), ('2016-04-05', 'vid461'), ('2016-04-05', 'vid386'), ('2016-04-05', 'vid468'), ('2016-04-05', 'vid356'), ('2016-04-05', 'vid434'), ('2016-04-05', 'vid493'), ('2016-04-05', 'vid403'), ('2016-04-05', 'vid469'), ('2016-04-05', 'vid473'), ('2016-04-05', 'vid433'), ('2016-04-05', 'vid425'), ('2016-04-05', 'vid449'), ('2016-04-05', 'vid459'), ('2016-04-05', 'vid378'), ('2016-04-05', 'vid389'), ('2016-04-05', 'vid367'), ('2016-04-05', 'vid479'), ('2016-04-05', 'vid485'), ('2016-04-05', 'vid474'), ('2016-04-05', 'vid406'), ('2016-04-05', 'vid456'), ('2016-04-05', 'vid491'), ('2016-04-05', 'vid475'), ('2016-04-05', 'vid431'), ('2016-04-05', 'vid432'), ('2016-04-05', 'vid375'), ('2016-04-05', 'vid444'), ('2016-04-05', 'vid428'), ('2016-04-05', 'vid357'), ('2016-04-05', 'vid355'), ('2016-04-05', 'vid480'), ('2016-04-05', 'vid419'), ('2016-04-05', 'vid368'), ('2016-04-05', 'vid371'), ('2016-04-05', 'vid380'), ('2016-04-05', 'vid488'), ('2016-04-05', 'vid393'), ('2016-04-05', 'vid353'), ('2016-04-05', 'vid442'), ('2016-04-05', 'vid467'), ('2016-04-05', 'vid443'), ('2016-04-05', 'vid362'), ('2016-04-05', 'vid423'), ('2016-04-05', 'vid422'), ('2016-04-05', 'vid391'), ('2016-04-05', 'vid463'), ('2016-04-05', 'vid381'), ('2016-04-05', 'vid404'), ('2016-04-05', 'vid476'), ('2016-04-05', 'vid435'), ('2016-04-05', 'vid396'), ('2016-04-05', 'vid349'), ('2016-04-05', 'vid397'), ('2016-04-05', 'vid401'), ('2016-04-05', 'vid402'), ('2016-04-05', 'vid430'), ('2016-04-05', 'vid390'), ('2016-04-05', 'vid370'), ('2016-04-05', 'vid412'), ('2016-04-05', 'vid385'), ('2016-04-05', 'vid439'), ('2016-04-05', 'vid410'), ('2016-04-05', 'vid464'), ('2016-04-05', 'vid363'), ('2016-04-05', 'vid470'), ('2016-04-05', 'vid354'), ('2016-04-05', 'vid417'), ('2016-04-05', 'vid452'), ('2016-04-05', 'vid472'), ('2016-04-05', 'vid440'), ('2016-04-05', 'vid482'), ('2016-04-05', 'vid478'), ('2016-04-05', 'vid447'), ('2016-04-05', 'vid477'), ('2016-04-05', 'vid413'), ('2016-04-05', 'vid379'), ('2016-04-05', 'vid445'), ('2016-04-05', 'vid492'), ('2016-04-05', 'vid366'), ('2016-04-05', 'vid427'), ('2016-04-05', 'vid420'), ('2016-04-05', 'vid414'), ('2016-04-05', 'vid426'), ('2016-04-05', 'vid407'), ('2016-04-05', 'vid415'), ('2016-04-05', 'vid398'), ('2016-04-05', 'vid471'), ('2016-04-05', 'vid457'), ('2016-04-05', 'vid384'), ('2016-04-05', 'vid373'), ('2016-04-05', 'vid462'), ('2016-04-05', 'vid418'), ('2016-04-05', 'vid408')]
        self.video_per_day['2016-04-06'] = [('2016-04-06', 'vid144'), ('2016-04-06', 'vid222'), ('2016-04-06', 'vid200'), ('2016-04-06', 'vid195'), ('2016-04-06', 'vid159'), ('2016-04-06', 'vid187'), ('2016-04-06', 'vid296'), ('2016-04-06', 'vid314'), ('2016-04-06', 'vid287'), ('2016-04-06', 'vid148'), ('2016-04-06', 'vid157'), ('2016-04-06', 'vid295'), ('2016-04-06', 'vid267'), ('2016-04-06', 'vid264'), ('2016-04-06', 'vid280'), ('2016-04-06', 'vid209'), ('2016-04-06', 'vid176'), ('2016-04-06', 'vid217'), ('2016-04-06', 'vid204'), ('2016-04-06', 'vid259'), ('2016-04-06', 'vid163'), ('2016-04-06', 'vid224'), ('2016-04-06', 'vid291'), ('2016-04-06', 'vid154'), ('2016-04-06', 'vid271'), ('2016-04-06', 'vid262'), ('2016-04-06', 'vid290'), ('2016-04-06', 'vid219'), ('2016-04-06', 'vid218'), ('2016-04-06', 'vid220'), ('2016-04-06', 'vid183'), ('2016-04-06', 'vid306'), ('2016-04-06', 'vid164'), ('2016-04-06', 'vid266'), ('2016-04-06', 'vid190'), ('2016-04-06', 'vid155'), ('2016-04-06', 'vid215'), ('2016-04-06', 'vid244'), ('2016-04-06', 'vid173'), ('2016-04-06', 'vid225'), ('2016-04-06', 'vid208'), ('2016-04-06', 'vid250'), ('2016-04-06', 'vid221'), ('2016-04-06', 'vid234'), ('2016-04-06', 'vid308'), ('2016-04-06', 'vid203'), ('2016-04-06', 'vid298'), ('2016-04-06', 'vid238'), ('2016-04-06', 'vid286'), ('2016-04-06', 'vid210'), ('2016-04-06', 'vid312'), ('2016-04-06', 'vid196'), ('2016-04-06', 'vid282'), ('2016-04-06', 'vid153'), ('2016-04-06', 'vid226'), ('2016-04-06', 'vid309'), ('2016-04-06', 'vid161'), ('2016-04-06', 'vid269'), ('2016-04-06', 'vid142'), ('2016-04-06', 'vid301'), ('2016-04-06', 'vid245'), ('2016-04-06', 'vid231'), ('2016-04-06', 'vid145'), ('2016-04-06', 'vid303'), ('2016-04-06', 'vid168'), ('2016-04-06', 'vid316'), ('2016-04-06', 'vid275'), ('2016-04-06', 'vid198'), ('2016-04-06', 'vid311'), ('2016-04-06', 'vid281'), ('2016-04-06', 'vid305'), ('2016-04-06', 'vid283'), ('2016-04-06', 'vid292'), ('2016-04-06', 'vid184'), ('2016-04-06', 'vid294'), ('2016-04-06', 'vid270'), ('2016-04-06', 'vid233'), ('2016-04-06', 'vid297'), ('2016-04-06', 'vid169'), ('2016-04-06', 'vid160'), ('2016-04-06', 'vid194'), ('2016-04-06', 'vid273'), ('2016-04-06', 'vid201'), ('2016-04-06', 'vid230'), ('2016-04-06', 'vid258'), ('2016-04-06', 'vid223'), ('2016-04-06', 'vid172'), ('2016-04-06', 'vid152'), ('2016-04-06', 'vid213'), ('2016-04-06', 'vid178'), ('2016-04-06', 'vid285'), ('2016-04-06', 'vid175'), ('2016-04-06', 'vid261'), ('2016-04-06', 'vid242'), ('2016-04-06', 'vid186'), ('2016-04-06', 'vid313'), ('2016-04-06', 'vid207'), ('2016-04-06', 'vid263'), ('2016-04-06', 'vid177'), ('2016-04-06', 'vid255'), ('2016-04-06', 'vid307'), ('2016-04-06', 'vid151'), ('2016-04-06', 'vid284'), ('2016-04-06', 'vid293'), ('2016-04-06', 'vid254'), ('2016-04-06', 'vid146'), ('2016-04-06', 'vid214'), ('2016-04-06', 'vid150'), ('2016-04-06', 'vid278'), ('2016-04-06', 'vid276'), ('2016-04-06', 'vid227'), ('2016-04-06', 'vid256'), ('2016-04-06', 'vid253'), ('2016-04-06', 'vid197'), ('2016-04-06', 'vid147'), ('2016-04-06', 'vid162'), ('2016-04-06', 'vid188'), ('2016-04-06', 'vid211'), ('2016-04-06', 'vid182'), ('2016-04-06', 'vid180'), ('2016-04-06', 'vid199'), ('2016-04-06', 'vid149'), ('2016-04-06', 'vid299'), ('2016-04-06', 'vid212'), ('2016-04-06', 'vid165'), ('2016-04-06', 'vid310'), ('2016-04-06', 'vid277'), ('2016-04-06', 'vid191'), ('2016-04-06', 'vid265'), ('2016-04-06', 'vid170'), ('2016-04-06', 'vid143'), ('2016-04-06', 'vid240'), ('2016-04-06', 'vid171'), ('2016-04-06', 'vid304'), ('2016-04-06', 'vid237'), ('2016-04-06', 'vid236'), ('2016-04-06', 'vid257'), ('2016-04-06', 'vid174'), ('2016-04-06', 'vid232'), ('2016-04-06', 'vid205'), ('2016-04-06', 'vid179'), ('2016-04-06', 'vid268'), ('2016-04-06', 'vid206'), ('2016-04-06', 'vid248'), ('2016-04-06', 'vid252'), ('2016-04-06', 'vid249'), ('2016-04-06', 'vid243'), ('2016-04-06', 'vid166'), ('2016-04-06', 'vid202'), ('2016-04-06', 'vid279'), ('2016-04-06', 'vid288'), ('2016-04-06', 'vid192'), ('2016-04-06', 'vid193'), ('2016-04-06', 'vid228'), ('2016-04-06', 'vid247'), ('2016-04-06', 'vid246'), ('2016-04-06', 'vid239'), ('2016-04-06', 'vid181'), ('2016-04-06', 'vid241'), ('2016-04-06', 'vid289'), ('2016-04-06', 'vid272'), ('2016-04-06', 'vid251'), ('2016-04-06', 'vid229'), ('2016-04-06', 'vid156'), ('2016-04-06', 'vid235'), ('2016-04-06', 'vid189'), ('2016-04-06', 'vid158'), ('2016-04-06', 'vid260'), ('2016-04-06', 'vid315'), ('2016-04-06', 'vid167'), ('2016-04-06', 'vid216'), ('2016-04-06', 'vid274'), ('2016-04-06', 'vid185'), ('2016-04-06', 'vid300'), ('2016-04-06', 'vid302')]
        self.video_per_day['2016-04-07'] = [('2016-04-07', 'vid15'), ('2016-04-07', 'vid33'), ('2016-04-07', 'vid19'), ('2016-04-07', 'vid11'), ('2016-04-07', 'vid24'), ('2016-04-07', 'vid9'), ('2016-04-07', 'vid21'), ('2016-04-07', 'vid5'), ('2016-04-07', 'vid14'), ('2016-04-07', 'vid7'), ('2016-04-07', 'vid26'), ('2016-04-07', 'vid16'), ('2016-04-07', 'vid13'), ('2016-04-07', 'vid28'), ('2016-04-07', 'vid36'), ('2016-04-07', 'vid3'), ('2016-04-07', 'vid35'), ('2016-04-07', 'vid22'), ('2016-04-07', 'vid2'), ('2016-04-07', 'vid18'), ('2016-04-07', 'vid30'), ('2016-04-07', 'vid12'), ('2016-04-07', 'vid34'), ('2016-04-07', 'vid4'), ('2016-04-07', 'vid25'), ('2016-04-07', 'vid27'), ('2016-04-07', 'vid8'), ('2016-04-07', 'vid23'), ('2016-04-07', 'vid1'), ('2016-04-07', 'vid32'), ('2016-04-07', 'vid20'), ('2016-04-07', 'vid29'), ('2016-04-07', 'vid6'), ('2016-04-07', 'vid17'), ('2016-04-07', 'vid31'), ('2016-04-07', 'vid10')]
        self.video_per_day['2016-04-08'] = [('2016-04-08', 'vid318'), ('2016-04-08', 'vid345'), ('2016-04-08', 'vid333'), ('2016-04-08', 'vid335'), ('2016-04-08', 'vid338'), ('2016-04-08', 'vid319'), ('2016-04-08', 'vid321'), ('2016-04-08', 'vid326'), ('2016-04-08', 'vid342'), ('2016-04-08', 'vid322'), ('2016-04-08', 'vid334'), ('2016-04-08', 'vid317'), ('2016-04-08', 'vid329'), ('2016-04-08', 'vid343'), ('2016-04-08', 'vid336'), ('2016-04-08', 'vid325'), ('2016-04-08', 'vid331'), ('2016-04-08', 'vid337'), ('2016-04-08', 'vid324'), ('2016-04-08', 'vid320'), ('2016-04-08', 'vid340'), ('2016-04-08', 'vid323'), ('2016-04-08', 'vid347'), ('2016-04-08', 'vid332'), ('2016-04-08', 'vid328'), ('2016-04-08', 'vid327'), ('2016-04-08', 'vid330'), ('2016-04-08', 'vid341'), ('2016-04-08', 'vid344'), ('2016-04-08', 'vid346'), ('2016-04-08', 'vid339')]
        self.video_per_day['2016-04-11'] = [('2016-04-11', 'vid68'), ('2016-04-11', 'vid51'), ('2016-04-11', 'vid114'), ('2016-04-11', 'vid103'), ('2016-04-11', 'vid123'), ('2016-04-11', 'vid131'), ('2016-04-11', 'vid83'), ('2016-04-11', 'vid100'), ('2016-04-11', 'vid87'), ('2016-04-11', 'vid109'), ('2016-04-11', 'vid62'), ('2016-04-11', 'vid91'), ('2016-04-11', 'vid49'), ('2016-04-11', 'vid55'), ('2016-04-11', 'vid129'), ('2016-04-11', 'vid115'), ('2016-04-11', 'vid82'), ('2016-04-11', 'vid64'), ('2016-04-11', 'vid96'), ('2016-04-11', 'vid77'), ('2016-04-11', 'vid107'), ('2016-04-11', 'vid38'), ('2016-04-11', 'vid71'), ('2016-04-11', 'vid90'), ('2016-04-11', 'vid125'), ('2016-04-11', 'vid88'), ('2016-04-11', 'vid127'), ('2016-04-11', 'vid58'), ('2016-04-11', 'vid133'), ('2016-04-11', 'vid139'), ('2016-04-11', 'vid69'), ('2016-04-11', 'vid97'), ('2016-04-11', 'vid93'), ('2016-04-11', 'vid92'), ('2016-04-11', 'vid116'), ('2016-04-11', 'vid67'), ('2016-04-11', 'vid95'), ('2016-04-11', 'vid104'), ('2016-04-11', 'vid66'), ('2016-04-11', 'vid86'), ('2016-04-11', 'vid40'), ('2016-04-11', 'vid126'), ('2016-04-11', 'vid48'), ('2016-04-11', 'vid120'), ('2016-04-11', 'vid72'), ('2016-04-11', 'vid105'), ('2016-04-11', 'vid44'), ('2016-04-11', 'vid50'), ('2016-04-11', 'vid47'), ('2016-04-11', 'vid122'), ('2016-04-11', 'vid45'), ('2016-04-11', 'vid79'), ('2016-04-11', 'vid46'), ('2016-04-11', 'vid94'), ('2016-04-11', 'vid117'), ('2016-04-11', 'vid59'), ('2016-04-11', 'vid137'), ('2016-04-11', 'vid128'), ('2016-04-11', 'vid53'), ('2016-04-11', 'vid54'), ('2016-04-11', 'vid63'), ('2016-04-11', 'vid132'), ('2016-04-11', 'vid61'), ('2016-04-11', 'vid119'), ('2016-04-11', 'vid56'), ('2016-04-11', 'vid65'), ('2016-04-11', 'vid41'), ('2016-04-11', 'vid111'), ('2016-04-11', 'vid124'), ('2016-04-11', 'vid73'), ('2016-04-11', 'vid39'), ('2016-04-11', 'vid57'), ('2016-04-11', 'vid84'), ('2016-04-11', 'vid52'), ('2016-04-11', 'vid136'), ('2016-04-11', 'vid81'), ('2016-04-11', 'vid76'), ('2016-04-11', 'vid141'), ('2016-04-11', 'vid121'), ('2016-04-11', 'vid140'), ('2016-04-11', 'vid102'), ('2016-04-11', 'vid113'), ('2016-04-11', 'vid80'), ('2016-04-11', 'vid70'), ('2016-04-11', 'vid98'), ('2016-04-11', 'vid85'), ('2016-04-11', 'vid138'), ('2016-04-11', 'vid78'), ('2016-04-11', 'vid60'), ('2016-04-11', 'vid110'), ('2016-04-11', 'vid37'), ('2016-04-11', 'vid130'), ('2016-04-11', 'vid43'), ('2016-04-11', 'vid75'), ('2016-04-11', 'vid99'), ('2016-04-11', 'vid112'), ('2016-04-11', 'vid108'), ('2016-04-11', 'vid134'), ('2016-04-11', 'vid135'), ('2016-04-11', 'vid42'), ('2016-04-11', 'vid101'), ('2016-04-11', 'vid74'), ('2016-04-11', 'vid89'), ('2016-04-11', 'vid106'), ('2016-04-11', 'vid118')]

        for date in ['2016-04-05','2016-04-06','2016-04-07','2016-04-08','2016-04-11']:
            for i in range(len(self.video_per_day[date])):
                A = self.video_per_day[date][i]
                self.video_per_day[date][i] = int(A[1].split('vid')[1])
            # print self.video_per_day[date]

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
                if "-" not in line2:
                    obj_ids = map(int, line2.split(' '))
                    for id in obj_ids:
                        if vals[id]>0:
                            if id not in self.all_objects:
                                self.all_objects.append(id)
                            self.objects[count2].append(id)
                count=0
                # print self.objects[count2]
                count2+=1
                # print '-----------'
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
        # what Majd said is ok
        self.GT_dict[21] = ['button', 'copier', 'copy', 'document', 'door', 'machine', 'paper', 'printer', 'screen','tray']
        self.GT_dict[22] = ['button', 'copier', 'copy', 'document', 'door', 'machine', 'paper', 'printer', 'screen','tray']
        self.GT_dict[23] = ['button', 'copier', 'copy', 'document', 'door', 'machine', 'paper', 'printer', 'screen','tray']
        self.GT_dict[24] = ['button', 'copier', 'copy', 'document', 'door', 'machine', 'paper', 'printer', 'screen','tray']

        self.GT_dict[19] = ['cooler', 'cup', 'dispenser', 'glass', 'water']

        self.GT_dict[13] = ['food', 'lunch', 'microwave']

        self.GT_dict[15] = ['beverage', 'box',  'can', 'coffee',  'container', 'creamer', 'cup',  'drink', 'milk', 'mug', 'spoon',  'sugar',  'tea', 'tea-bag', 'tea-coffee']

        self.GT_dict[14]  = ['fire_extenguisher']

        self.GT_dict[4]  = ['liquid',  'sink', 'tap', 'utensil', 'water']
        self.GT_dict[17] = ['liquid',  'sink', 'tap', 'utensil', 'water']
        self.GT_dict[18] = ['liquid',  'sink', 'tap', 'utensil', 'water']

        self.GT_dict[11] = ['bin', 'dustbin',  'garbage', 'trash', 'trashcan', 'waste']

        self.GT_dict[12] = ['bin', 'dustbin',  'garbage', 'trash', 'trashcan', 'waste']

        self.GT_dict[3] = ['fridge','refrigerator',"milk"]


        self.GT_dict[16] = ['kettle']

        self.mentioned_words = {}
        for i in range(1,494):
            for obj in self.objects[i]:
                if obj not in self.mentioned_words:
                    self.mentioned_words[obj] = []
                for word in self.objects_words[i]:
                    if word not in self.mentioned_words[obj]:
                        self.mentioned_words[obj].append(word)
            # print self.mentioned_words
            # print '---------'
        for obj in self.mentioned_words:
            self.GT_dict[obj] = list(set(self.mentioned_words[obj]).intersection(self.GT_dict[obj]))

        self.GT_total_links = 0
        for i in self.GT_dict:
            self.GT_total_links += len(self.GT_dict[i])

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

    def _assignment_matrix(self,date):
        self.all_objects_words = copy.copy(self.all_objects_words_copy)
        self.objects_words_count = {}
        self.cluster_count = {}
        # stop = len(self.video_num)*fraction
        count = 0
        videos_seen = {}
        for vid in self.video_num:
            if vid in self.video_per_day[date]:
                self.ok_videos.append(vid)
                for cluster in self.objects[vid]:
                    self.ok_clusters.append(cluster)
            videos_seen[vid] = {}
            videos_seen[vid]['clusters'] = []
            videos_seen[vid]['object'] = []
        for vid in self.ok_videos:
            for cluster in self.objects[vid]:
                if cluster in videos_seen[vid]['clusters']:
                    continue
                videos_seen[vid]['clusters'].append(cluster)
                if cluster not in self.cluster_count:
                    self.cluster_count[cluster] = 0
                # if count <= stop:
                if vid in self.ok_videos:
                    self.cluster_count[cluster] += 1
                for name in self.objects_words[vid]:
                    if name in self.all_objects_words:
                        noun_i = self.all_objects_words.index(name)
                        if noun_i not in self.objects_words_count:
                            self.objects_words_count[noun_i] = 0
                        # if count <= stop:
                        if vid in self.ok_videos:
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
            if self.objects_words_count[i]<10:
                objects_words_to_remove.append(i)
        for i in reversed(objects_words_to_remove):
            # print '>>', self.all_objects_words[i]
            del self.all_objects_words[i]

        self.CM_nouns = np.zeros((len(self.all_objects_words),len(self.all_objects)))
        self.CM_clust = np.zeros((len(self.all_objects),len(self.all_objects_words)))
        self.objects_words_count = {}
        self.cluster_count = {}
        # stop = len(self.video_num)*fraction
        count = 0
        videos_seen = {}
        for cluster,vid in zip(self.objects,self.video_num):
            videos_seen[vid] = {}
            videos_seen[vid]['clusters'] = []
            videos_seen[vid]['object'] = []

        for vid in self.ok_videos:
            for cluster_name in self.objects[vid]:
                cluster = self.all_objects.index(cluster_name)
                if cluster in videos_seen[vid]['clusters']:
                    continue
                videos_seen[vid]['clusters'].append(cluster)
                if cluster not in self.cluster_count:
                    self.cluster_count[cluster] = 0
                # if count <= stop:
                if vid in self.ok_videos:
                    self.cluster_count[cluster] += 1
                for name in self.objects_words[vid]:
                    # if name == 'printing':
                    #     print '>>>>>>>>',vid
                    #     print self.objects[vid]
                    if name in self.all_objects_words:
                        noun_i = self.all_objects_words.index(name)
                        if noun_i not in self.objects_words_count:
                            self.objects_words_count[noun_i] = 0
                        if vid in self.ok_videos:
                        # if count <= stop:
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
                    if assignment[1] in self.GT_dict[self.all_objects[assignment[0]]]:
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
        # print max_assignments
        print self.f_score
        # print '-----------'
        pickle.dump( self.f_score, open( self.dir2+'objects_incremental.p', "wb" ) )
        print self.objects[235]
        # pickle.dump( self.f_score, open( self.dir_faces+'faces_f_score3.p', "wb" ) )

    def _plot_incremental(self):
        self.f_score = pickle.load(open(self.dir2+'objects_incremental.p',"rb"))
        x = np.arange(len(self.f_score))/float(5-1)*493
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
    f._get_video_per_days()
    f._read_objects()
    f._read_tags()
    # f._cluster_objects()
    # f._read_objects_clusters()
    # f._plot_objects_clusters()
    f._get_groundTruth()
    # f._assignment_matrix("2016-04-05")
    # f._LP_assign(.05)
    # f._compute_measures()

    for i,date in enumerate(['2016-04-05','2016-04-06','2016-04-07','2016-04-08','2016-04-11']):
        f._assignment_matrix(date)
        f._LP_assign(.07)
    f._plot_incremental()
    # f.max = 10
    # for i in range(1,f.max+1):
    #     f._assignment_matrix(i/float(f.max))
    #     f._LP_assign(.05)
    # f._plot_incremental()
    # # f._plot_f_score()
    # f._pretty_plot()

if __name__=="__main__":
    main()
