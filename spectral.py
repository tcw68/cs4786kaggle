import numpy as np
from scipy import linalg as LA
from scipy.cluster.vq import kmeans, vq
from collections import deque
import math
from sklearn import cluster

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


# def spectralClustering(M, K):
# 	print "Finding normalized D"
# 	normD = np.zeros((6000, 6000))

# 	for i in range(M.shape[0]):
# 		normD[i,i] = (float)(1/math.sqrt(sum(M[i,])))
# 	print "Done finding normalized D"

# 	print "Finding normalized L"
# 	normL = np.subtract(np.identity(6000), np.dot(np.dot(normD, M), normD))
# 	print "Done finding normalized L"

# 	print "Finding eigenvalues and eigenvectors"
# 	evalues, vr = LA.eig(normL)
# 	print "Done finding eigenvalues and eigenvectors"
	
# 	print "Finding the smallest K eigenvectors"
# 	np_vectors = np.array(vr[:K])
# 	np_vectors = np_vectors.T
# 	print "Done finding the smallest K eigenvectors"

# 	print "Performing K-Means Clustering"
# 	codebook, _ = kmeans(np_vectors, 10)
# 	code, _ = vq(vectors, codebook)
# 	clusters = [[],[],[],[],[],[],[],[],[],[]]
# 	for i in range(code.shape[0]):
# 		clusters[code[i]].append()
# 	print "Done performing K-Means Clustering"


def seedDict(seedFile):
	print "Filling in dictionary..."
	seed = {}
	for line in seedFile:
		(node1, node2) = [int(x) for x in line.split(',')]
		seed[node1] = node2
	print "Done filling in dictionary"
	return seed

# Run the program
def run():
	print "Loading files..."
	featuresVectorFile = open('../Extracted_features.csv', 'r')
	seedFile = open('../Seed.csv', 'r')
	graphFile = open('../Graph.csv', 'r')
	print 'Done loading files'

	np.set_printoptions(threshold=np.nan)
	M = numpyMatrix(graphFile)
	seed = seedDict(seedFile)


	# spectralClustering(M, 1000)
	# features = np.genfromtxt('Extracted_features.csv', delimiter = ',', dtype='int32')
	# spectral = cluster.SpectralClustering(n_clusters = 10, eigen_solver = 'arpack', affinity = "nearest_neighbors")
	# spectral.fit(features[6001:,])
	# y_pred = spectral.labels_.astype(np.int)


	similarityMatrix = cluster.SpectralClustering(n_clusters = 10, eigen_solver = 'arpack', affinity = "precomputed")
	similarityMatrix.fit(M)
	y_labels = similarityMatrix.labels_.astype(np.int)

	test = {}
	for (node, digit) in seed.items():
		test[node] = (y_labels[node-1], digit)

	print "test", test.items()

run()
