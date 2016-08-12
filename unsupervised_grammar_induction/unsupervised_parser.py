from soegaard import *
from conll import *

#parse = parse_sentence(['pick','up','the','yellow','prism','above','the','red','cube'], [], False)
#[(0, 4), (2, 1), (4, 2), (4, 3), (4, 5), (4, 6), (6, 8), (6, 7), (8, 9)]

parse = parse_sentence(['move','the','yellow','prism','above','the','red','cube'], [], True, ['VB','DT','JJ','NN','IN','DT','JJ','NN'], 'en-ptb')
# print parse.nodes()
# print parse.edges()
# print export_to_conll_format(parse)

#pick/VB up/RP the/DT yellow/JJ prism/NN above/IN the/DT red/JJ cube/NN
# print parse
