import pulp
import cPickle as pickle
import sys

dir1 = "/home/omari/Datasets/ECAI_dataset/faces/"
CM_nouns, CM_clust, nouns_count, cluster_count, all_nouns = pickle.load(open( dir1 + "faces_correlation.p", "rb" ) )

faces = range(33)
words=all_nouns

if len(sys.argv)>1:
    max_assignments=int(sys.argv[1])
else:
    max_assignments=int(len(faces)*2)


def word_strength(face, word):
    #conditional probabiltiy: (N(w,f)/N(f) + N(w,f)/N(w)) /2
    # return round((100.0*CM_nouns[words.index(word)][face]/cluster_count[face] + 100.0*CM_nouns[words.index(word)][face]/nouns_count[words.index(word)])/2)
    return round(max(100.0*CM_nouns[words.index(word)][face]/cluster_count[face] , 100.0*CM_nouns[words.index(word)][face]/nouns_count[words.index(word)]))


possible_assignments = [(x,y) for x in faces for y in words]

#create a binary variable for assignments
x = pulp.LpVariable.dicts('x', possible_assignments,
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)
# print x

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

print ([sum([pulp.value(x[assignment]) for assignment in possible_assignments if face==assignment[0] ]) for face in faces])

f = open(dir1+"circos/faces.txt")
for assignment in possible_assignments:
    if x[assignment].value() == 1.0:
        print(assignment)
