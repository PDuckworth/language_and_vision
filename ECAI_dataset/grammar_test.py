import os
from nltk.parse import stanford
os.environ['STANFORD_PARSER'] = '/home/omari/Datasets/ECAI_dataset/stanford-parser-full-2016-10-31'
os.environ['STANFORD_MODELS'] = '/home/omari/Datasets/ECAI_dataset/stanford-parser-full-2016-10-31'

# parser = stanford.StanfordParser(model_path="/location/of/the/englishPCFG.ser.gz")
parser = stanford.StanfordParser(model_path="/home/omari/Datasets/ECAI_dataset/stanford-parser-full-2016-10-31/jar/englishPCFG.ser.gz")
sentences = parser.raw_parse_sents(("Andy is tall and is wearing a blue shirt with black shorts", "What is your name?"))
# print sentences

# GUI
for line in sentences:
    for sentence in line:
        print sentence
        sentence.draw()
