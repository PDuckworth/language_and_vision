import os
import nltk
from nltk.parse import stanford
import getpass
import pickle

class grammar():
    """docstring for grammar"""
    def __init__(self):
        self.username = getpass.getuser()
        self.dir1 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/features/vid'
        self.dir2 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/grammar/'
        self.dir_text = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/ECAI_annotations/vid'
        self.folder = 1
        self.raw_sentences = {}

        #grammar stuff
        os.environ['STANFORD_PARSER'] = '/home/omari/Datasets_old/ECAI_dataset_segmented/stanford-parser-full-2016-10-31'
        os.environ['STANFORD_MODELS'] = '/home/omari/Datasets_old/ECAI_dataset_segmented/stanford-parser-full-2016-10-31'
        self.parser = stanford.StanfordParser(model_path="/home/omari/Datasets_old/ECAI_dataset_segmented/stanford-parser-full-2016-10-31/jar/englishPCFG.ser.gz")
        self.start = 1
        self.end = 494
        self.tags = {}

    def _read_annotations(self):
        self.sentences = {}
        self.words = {}
        self.words_count = {}
        self.all_words = []
        self.all_passed_words = []
        for i in range(self.start,self.end):
            self.sentences[i] = {}
            self.raw_sentences[i] = []
            self.words[i] = []
            f = open(self.dir_text+str(i)+'/person.txt','r')
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
            self.tags[i]['upper_garment'] = {}
            self.tags[i]['lower_garment'] = {}
            self.tags[i]['name'] = {}
            # Parsed = self.parser.raw_parse_sents(self.raw_sentences[i])
            for c1,sentence in enumerate(self.sentences[i]):
                tree = pickle.load(open( self.dir2+"/trees/tree_"+str(i)+'_'+str(c1)+".p", "rb" ))
                # print self.sentences[i][sentence]
                # print tree
                for subtree in tree.subtrees():
                    #sub trees
                    if subtree.height() == 3:
                        if subtree.label() == 'NP':
                            ok1 = 0
                            for word in subtree.leaves():
                                if word in self.upper_garment:
                                    ok1 = 1
                            ok2 = 0
                            for subsubtree in subtree.subtrees():
                                if subsubtree.label() == 'JJ':
                                    ok2 =1
                            if ok1 and ok2:
                                print subtree
                                for subsubtree in subtree.subtrees():
                                    if subsubtree.label()=='JJ':
                                        leaf = subsubtree.leaves()[0]
                                        if leaf not in self.tags[i]['upper_garment']:
                                            self.tags[i]['upper_garment'][leaf] = 0
                                        self.tags[i]['upper_garment'][leaf] += 1

                            ok1 = 0
                            for word in subtree.leaves():
                                if word in self.lower_garment:
                                    ok1 = 1
                            if ok1 and ok2:
                                print subtree
                                for subsubtree in subtree.subtrees():
                                    if subsubtree.label()=='JJ':
                                        leaf = subsubtree.leaves()[0]
                                        if leaf not in self.tags[i]['lower_garment']:
                                            self.tags[i]['lower_garment'][leaf] = 0
                                        self.tags[i]['lower_garment'][leaf] += 1
                            ok1 = 0
                            for subsubtree in subtree.subtrees():
                                if subsubtree.label() == 'NN':
                                    ok1 =1
                            if ok1 and not ok2:
                                print subtree
                                for subsubtree in subtree.subtrees():
                                    if subsubtree.label()=='NN':
                                        leaf = subsubtree.leaves()[0]
                                        if leaf not in self.tags[i]['name']:
                                            self.tags[i]['name'][leaf] = 0
                                        self.tags[i]['name'][leaf] += 1


    def _parse(self):
        for i in range(self.start,self.end):
            self.tags[i] = {}
            self.tags[i]['upper_garment'] = {}
            Parsed = self.parser.raw_parse_sents(self.raw_sentences[i])
            for c1,sentence in enumerate(Parsed):
                for c2,tree in enumerate(sentence):
                    pickle.dump( tree, open( self.dir2+"/trees/tree_"+str(i)+'_'+str(c1)+".p", "wb" ) )
                    for subtree in tree.subtrees():
                        #sub trees
                        if subtree.height() == 3:
                            if subtree.label() == 'NP':
                                ## upper garment
                                ok1 = 0
                                for word in subtree.leaves():
                                    if word in self.upper_garment:
                                        ok1 = 1
                                ok2 = 0
                                for subsubtree in subtree.subtrees():
                                    if subsubtree.label() == 'JJ':
                                        ok2 =1
                                if ok1 and ok2:
                                    print subtree
                                    for subsubtree in subtree.subtrees():
                                        if subsubtree.label()=='JJ':
                                            leaf = subsubtree.leaves()[0]
                                            if leaf not in self.tags[i]['upper_garment']:
                                                self.tags[i]['upper_garment'][leaf] = 0
                                            self.tags[i]['upper_garment'][leaf] += 1

                                ## lower garment

                    print '---',i
                    # tree.draw()

    def _save_data(self):
        pickle.dump( [self.tags,self.words_count], open( self.dir2+"tags.p", "wb" ) )

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
    f._read_garments()
    # f._pos_tag()
    f._read_parse()
    # f._parse()
    f._save_data()
    f._print_results()

if __name__=="__main__":
    main()



# print 'test'
# # GUI
# for line in Parsed:
#     for sentence in line:
#         print sentence
        # sentence.draw()
