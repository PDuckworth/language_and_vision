
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from networkx import *
import numpy as np

Dir = '/home/omari/Desktop/Python/language/Simultaneous_learning_and_ground/images2/'
#-------------------------------------------------------------------------------------#
def make_order_graph(hyp,f_order):
	
	#to make the position

	# actual world graph
	plt.figure(2)
        plt.cla()
	#graph_maker_order(hyp['word_order'],hyp['order'])

	plt.axis('off')
    	plt.draw()

	G = nx.DiGraph()
	for i in range(len(hyp['word_order'])):
		for j in range(len(hyp['word_order'])):
			if i!=j and hyp['order'][i,j]!=0:
				G.add_edges_from([(hyp['word_order'][i], hyp['word_order'][j])], weight=1/hyp['order'][i,j])

	val_map = {}
	for j in hyp['valid_HSV_hyp']:
		val_map[j] = 1.0

	for j in hyp['valid_dir_hyp']:
		val_map[j] = .7

	#val_map = {'red': 1.0,'blue': 0.5714285714285714,'right': 0.0}

	values = [val_map.get(node, 0.45) for node in G.nodes()]

	edge_labels=dict([((u,v,),d['weight'])
		         for u,v,d in G.edges(data=True)])
	red_edges = []
	edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]

	pos=nx.spring_layout(G)
	nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
	nx.draw(G,pos, node_color = values, node_size=1500,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
	plt.draw()

	o = hyp['word_order']
	#for S in hyp['objects_hyp']:
		#hyp['valid_dis_hyp']
		#for T in hyp['valid_dir_hyp']:
			#S = j
			#T = k
			#print nx.shortest_path(G,source=S,target=T)
			#print nx.shortest_path(G,source=S,target=T,weight='weight')

	if hyp['objects_hyp'][0] != '':
		for j in range(len(hyp['objects_hyp'])-1):
			for k in range(j+1,len(hyp['objects_hyp'])):
				S = hyp['objects_hyp'][j]
				T = hyp['objects_hyp'][k]
				#print S,T
				#if S!=T:
					#print nx.shortest_path(G,source=S,target=T)
					#print nx.shortest_path(G,source=S,target=T,weight='weight')
		
#-------------------------------------------------------------------------------------#
def graph_maker_order(o,e):
	labels={}
	G = Graph()
	G.clear()	

	for i in range(len(o)):
		G.add_node(i,color='c',type1=o[i],type2='')
		labels[i] = o[i]
		#object_nodes_counter += 1
	
	for i in range(len(o)):
		for j in range(len(o)):
			if i!=j and e[i,j]!=0:
				G.add_edge(i,j,width=e[i,j]/e.max()*3,w=1/e[i,j])

	if 'red' in o and 'right' in o:
		print nx.shortest_path(G,source=o.index('red'),target=o.index('right'),weight='w')
		results = []
		for j in nx.shortest_path(G,source=o.index('red'),target=o.index('right'),weight='w'):
			results.append(o[j])
		print 'red ',results,' right'
    	update_graph_order(G,labels)						# Graph generation
	plt.savefig(Dir+"word_order.png") # or .pdf
	#plt.clf()

	return G

#-------------------------------------------------------------------------------------#
def update_graph_order(G,labels):
	tmp = get_node_attributes(G,'color')
	A = G.nodes()
	color = []
	for i in A:
		color = np.append(color,tmp[i])

	tmp = get_edge_attributes(G,'width')
	A = G.edges()
	w = []
	for i in A:
		w.append(tmp[i])

	#pos = get_node_attributes(G,'pos')
	pos=nx.spring_layout(G)
	draw_networkx_nodes(G,pos=pos,node_color=color,node_size=1200)
	#draw_networkx_nodes(G,pos)
	draw_networkx_edges(G,pos=pos,width=w,alpha=1)
	draw_networkx_labels(G,pos,labels,font_size=10)

#-------------------------------------------------------------------------------------#
def Update_world_graph(o_color,o_qsr,f_world):
	o1 = []
	spatial = []

	for p in o_color['color']:
		o1.append(str(p))


	A = o_qsr['obj_number']-1

	if A == 1:
		s = o_qsr[str(0)+'-'+str(1)+'-dis']
		a = o_qsr[str(0)+'-'+str(1)+'-ang']
		spatial.append([s,a])

	else:
		for p in range(A):
			for k in range(p+1,A+1):
				s = o_qsr[str(p)+'-'+str(k)+'-dis']
				a = o_qsr[str(p)+'-'+str(k)+'-ang']
				spatial.append([s,a])

	# actual world graph
	plt.figure(1)
        plt.cla()
	graph_maker(o1,spatial,1)
	plt.axis('off')
    	plt.draw()

#-------------------------------------------------------------------------------------#
def grapelet(g,obj1,spatial,obj2):
	global spatial_nodes_counter,labels
	g.add_node(spatial_nodes_counter,pos=(spatial_nodes_counter-object_nodes_counter,3),color='g',type1=spatial,type2='spatial')
	labels[spatial_nodes_counter] = spatial
	g.add_edge(spatial_nodes_counter,obj1,dirx='')
	g.add_edge(spatial_nodes_counter,obj2,dirx='')
	spatial_nodes_counter = spatial_nodes_counter+1

#-------------------------------------------------------------------------------------#
def grapelet_temporal(g,spa1,temporal,spa2):
	global temporal_nodes_counter,labels,spatial_nodes_counter
	g.add_node(temporal_nodes_counter,pos=(temporal_nodes_counter-spatial_nodes_counter,5),color='r',type1=temporal,type2='temporal')
	labels[temporal_nodes_counter] = temporal
	g.add_edge(temporal_nodes_counter,spa1,dirx='from '+G.node[spa1]['type1'])
	g.add_edge(temporal_nodes_counter,spa2,dirx='to '+G.node[spa2]['type1'])
	temporal_nodes_counter = temporal_nodes_counter+1

#-------------------------------------------------------------------------------------#
def update_graph():
	global G,labels
	tmp = get_node_attributes(G,'color')
	A = G.nodes()
	color = []
	for i in A:
		color = np.append(color,tmp[i])
	pos = get_node_attributes(G,'pos')
	draw_networkx_nodes(G,pos,node_size=[1000+2000/(spatial_nodes_counter+1)],node_color=color)
	#draw_networkx_nodes(G,pos)
	draw_networkx_edges(G,pos,width=1.0,alpha=1)
	draw_networkx_labels(G,pos,labels,font_size=10)

#-------------------------------------------------------------------------------------#
def graph_maker(o1,spatial,plotting):
    	global G,object_nodes_counter,spatial_nodes_counter,temporal_nodes_counter,labels,Main_counter

    	object_nodes_counter = 0
    	spatial_nodes_counter = 0
	temporal_nodes_counter = 0
	labels={}
	G = Graph()
	G.clear()	

	color = []
	for i in range(len(o1)):
		G.add_node(i,pos=(object_nodes_counter,1),color='c',type1=o1[i],type2='object')
		labels[i] = o1[i]
		object_nodes_counter += 1	
    	spatial_nodes_counter = object_nodes_counter

	#if len(spatial)==1:
	#	grapelet(G,0,spatial[0],1)		# QSR to Graph

	#else:
	counter = 0
	for j in range(len(o1)-1):
		for k in range(j+1,len(o1)):
			grapelet(G,j,spatial[counter],k)		# QSR to Graph
			counter += 1
	"""
	counter = 0
	temporal_nodes_counter = spatial_nodes_counter
	for i in range(len(spatial)-1):
	    for j in range(i+1,len(spatial)):
		grapelet_temporal(G,i+object_nodes_counter,temporal[counter],j+object_nodes_counter)		# QSR to Graph
		counter += 1
	"""
    	if plotting:
		update_graph()						# Graph generation
	#plt.savefig("/home/omari/Desktop/Python/language/graph"+str(Main_counter)+".png") # or .pdf
	#plt.clf()

	return G


