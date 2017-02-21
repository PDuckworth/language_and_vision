import os
import nltk
from nltk.parse import stanford
import getpass
import pickle

class grammar():
    """docstring for grammar"""
    def __init__(self):
        self.username = getpass.getuser()
        self.dir1 = '/home/'+self.username+'/Datasets/ECAI_dataset/features/vid'
        self.dir2 = '/home/'+self.username+'/Datasets/ECAI_dataset/grammar/'
        self.dir_text = '/home/'+self.username+'/Datasets/ECAI_dataset/ECAI_annotations/vid'
        self.folder = 1
        self.raw_sentences = {}

        #grammar stuff
        os.environ['STANFORD_PARSER'] = '/home/omari/Datasets/ECAI_dataset/stanford-parser-full-2016-10-31'
        os.environ['STANFORD_MODELS'] = '/home/omari/Datasets/ECAI_dataset/stanford-parser-full-2016-10-31'
        self.parser = stanford.StanfordParser(model_path="/home/omari/Datasets/ECAI_dataset/stanford-parser-full-2016-10-31/jar/englishPCFG.ser.gz")
        self.start = 1
        self.end = 494
        self.list = [1,10,100,200,300]
        self.tags = {}

    def _read_annotations(self):
        self.sentences = {}
        self.words = {}
        self.words_count = {}
        self.all_words = []
        self.all_passed_words = []
        for i in range(self.start,self.end):
        # for i in self.list:
            print '>>>',i
            self.sentences[i] = {}
            self.raw_sentences[i] = []
            self.words[i] = []
            f = open(self.dir_text+str(i)+'/activity.txt','r')
            for count,line in enumerate(f):
                if count == 0 or "(X)" in line or line == "\n":
                    continue
                line = line.split('\n')[0]
                if '#' in line:
                    print line
                    sys.exit(1)
                line = line.lower()
                line = line.replace('.','')
                line = line.replace(',','')
                line = line.replace('/','-')
                line = line.replace('iis','is')
                line = line.replace('daniels','daniel')
                line = line.replace('allan','alan')
                line = line.replace('jenas','jeans')
                line = line.replace('suasan','susan')
                line = line.replace(' t ',' ')
                self.sentences[i][count] = line

                print i,line
                self.raw_sentences[i].append(line)

                for word in line.split(' '):
                    if word != '' and word not in self.words[i]:
                        self.words[i].append(word)
                    if word != '' and word not in self.all_words:
                        self.all_words.append(word)
                    if word != '':
                        if word not in self.words_count:
                            self.words_count[word] = 0
                        self.words_count[word] += 1

    def _read_garments(self):
        g = open( self.dir2+"upper_garment.txt", "r")
        self.upper_garment = []
        for word in g:
            self.upper_garment.append(word.split('\n')[0])

        g = open( self.dir2+"lower_garment.txt", "r")
        self.lower_garment = []
        for word in g:
            self.lower_garment.append(word.split('\n')[0])

    def _read_parse(self):
        for i in range(self.start,self.end):
            self.tags[i] = {}
            self.tags[i]['verb'] = {}
            self.tags[i]['object'] = {}
            for c1,sentence in enumerate(self.sentences[i]):
                tree = pickle.load(open( self.dir2+"/trees_activity/tree_"+str(i)+'_'+str(c1)+".p", "rb" ))
                # print tree
                for subtree in tree.subtrees():
                    # if subtree.height() == 0:
                    if subtree.label() in ['VBG','VBZ']:
                        for word in subtree.leaves():
                            word = str(word)
                            if word not in ["is","'s","has"]:
                                if word not in self.tags[i]['verb']:
                                    self.tags[i]['verb'][word] = 0
                                self.tags[i]['verb'][word] += 1

                    if subtree.label() in ['NN']:
                        for word in subtree.leaves():
                            word = str(word)
                            if word not in self.tags[i]['object']:
                                self.tags[i]['object'][word] = 0
                            self.tags[i]['object'][word] += 1
                #
                #             ok1 = 0
                #             for word in subtree.leaves():
                #                 if word in self.lower_garment:
                #                     ok1 = 1
                #             if ok1 and ok2:
                #                 print subtree
                #                 for subsubtree in subtree.subtrees():
                #                     if subsubtree.label()=='JJ':
                #                         leaf = subsubtree.leaves()[0]
                #                         if leaf not in self.tags[i]['lower_garment']:
                #                             self.tags[i]['lower_garment'][leaf] = 0
                #                         self.tags[i]['lower_garment'][leaf] += 1
                #             ok1 = 0
                #             for subsubtree in subtree.subtrees():
                #                 if subsubtree.label() in ['NN','NNS']:
                #                     ok1 =1
                #             if ok1 and not ok2:
                #                 print subtree
                #                 for subsubtree in subtree.subtrees():
                #                     if subsubtree.label() in ['NN','NNS']:
                #                         leaf = subsubtree.leaves()[0]
                #                         if leaf not in self.tags[i]['name']:
                #                             self.tags[i]['name'][leaf] = 0
                #                         self.tags[i]['name'][leaf] += 1

    def _parse(self):
        for i in range(self.start,self.end):
        # for i in self.list:
            self.tags[i] = {}
            self.tags[i]['verb'] = {}
            print '>>>',self.raw_sentences[i]
            Parsed = self.parser.raw_parse_sents(self.raw_sentences[i])
            print i
            for c1,sentence in enumerate(Parsed):
                for c2,tree in enumerate(sentence):
                    # print tree
                    pickle.dump( tree, open( self.dir2+"/trees_activity/tree_"+str(i)+'_'+str(c1)+".p", "wb" ) )

    def _save_data(self):
        pickle.dump( [self.tags,self.words_count], open( self.dir2+"tags_activity.p", "wb" ) )

    def _print_results(self):
        for i in self.tags:
            print '-------------',i
            for label in self.tags[i]:
                print '###',label,':'
                for word in self.tags[i][label]:
                    print word,':',self.tags[i][label][word],self.words_count[word]

def main():
    f = grammar()
    f._read_annotations()
    # f._read_garments()
    # f._pos_tag()
    ### swap between the next two commands: _parse save the trees, _read_parse is quicker and create the tags
    f._read_parse()
    # f._parse()
    f._save_data()
    # f._print_results()

if __name__=="__main__":
    main()



# print 'test'
# # GUI
# for line in Parsed:
#     for sentence in line:
#         print sentence
        # sentence.draw()
