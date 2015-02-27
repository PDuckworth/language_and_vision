
import numpy as np
import math 


#-------------------------------------------------------------------------------------#
def word_order_2(hyp,sentence):

	words = []
	print hyp['valid_HSV_hyp']
	print hyp['valid_dir_hyp']
	print hyp['valid_dis_hyp']

	s = ''
	for j in sentence.split(' '):
		if j in hyp['valid_HSV_hyp']:
			s+='HSV '
		elif j in hyp['valid_dis_hyp']:
			s+='dis '
		elif j in hyp['valid_dir_hyp']:
			s+='dir '
		else:
			s+='_ '
		words.append(j)
	print s
	return hyp

#-------------------------------------------------------------------------------------#
def word_order(hyp,sentence):
	#------------- create a list of all words ---------------#
	words = []
	for j in sentence.split(' '):
		words.append(j)
		if j not in hyp['word_order']:
			hyp['word_order'].append(j)

			A = len(hyp['word_order'])
			if A == 1: 	#initiate the order matrix
				hyp['order'] = np.zeros((1,1))
			else:		# add a row and a column to the word order matrix
				B = np.zeros((A,A))
				B[:A-1,:A-1] = hyp['order']
				hyp['order'] = B

	for k in range(len(words)):
		A = hyp['word_order'].index(words[k])
		hyp['order'][A,A] += 1			# diagonal serve as counter

	for k in range(len(words)-1):
		A = hyp['word_order'].index(words[k])
		B = hyp['word_order'].index(words[k+1])
		hyp['order'][A,B] += 1

	return hyp



#-------------------------------------------------------------------------------------#
def word_order_consistency(hyp):
	
	hyp['valid_word_order'] = []
	for i in range(len(hyp['word_order'])):
		for j in range(len(hyp['word_order'])):
			if i != j:
				counter = hyp['order'][i,i]
				word = hyp['order'][i,j]
				if word*1.0 >= counter*.9:
					hyp['valid_word_order'].append(hyp['word_order'][i]+' '+hyp['word_order'][j])

	#print hyp['valid_word_order']
	return hyp














