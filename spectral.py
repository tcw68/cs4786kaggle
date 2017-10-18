import numpy as np
from scipy import linalg as LA
from scipy.cluster.vq import kmeans, vq
from collections import deque
import math

def numpyMatrix(graphFile):
	print "Creating numpy adjacency matrix..."
	M = np.zeros((6000,6000), dtype='int32')

	print "Filling in matrix..."
	for line in graphFile:
		(node1, node2) = [int(x) for x in line.split(',')]
		M[node1-1][node2-1] = 1
		M[node2-1][node1-1] = 1

	graphFile.close()
	print "Done creating matrix"
	return M


def spectralClustering(M, K):
	normD = np.zeros((6000, 6000))

	for i in range(M.shape[0]):
		normD[i,i] = (float)(1/math.sqrt(sum(M[i,])))

	normL = np.subtract(np.identity(6000), np.dot(np.dot(normD, M), normD)
	evalues, _, vr = LA.eig(normL)
	vectors = []	


	order = np.argsort(evalues)
	for i in range(K):
		vectors.append(vr[order[i]])

	np_vectors = np.array(vectors)

	codebook, _ = kmeans(np_vectors, 10)
	code, _ = vq(vectors, codebook)


	clusters = [[],[],[],[],[],[],[],[],[],[]]
	for i in range(code.shape[0]):
		clusters[code[i]].append()





# Run the program
def run():
	print "Loading files..."
	featuresVectorFile = open('../Extracted_features.csv', 'r')
	seedFile = open('../Seed.csv', 'r')
	graphFile = open('../Graph.csv', 'r')
	print 'Done loading files'

	M = numpyMatrix(graphFile)

	spectralClustering(M)


run()
