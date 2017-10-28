import sys
import numpy as np
from scipy import linalg as LA
from collections import deque
import math
from sklearn import cluster
from sklearn.cluster import SpectralClustering, KMeans

# Create distance matrix 
# 0 = identical, larger number = greater dissimilarity, infinity = not similar at all
def findWeightedMatrix(graphFile, features):
	print "Creating numpy adjacency matrix..."
	M = np.full((6000,6000), float(100))

	print "Filling in matrix..."
	for line in graphFile:
		(node1, node2) = [int(x) for x in line.split(',')]
		euclidean_distance = np.linalg.norm(features[node1-1] - features[node2-1])
		M[node1-1][node2-1] = euclidean_distance
		M[node2-1][node1-1] = euclidean_distance

	# Set diagonal to 0 for identical nodes
	for i in range(M.shape[0]):
		M[i,i] = 0

	graphFile.close()
	print "Done creating matrix"
	return M

# Custom spectral clustering code
def spectralClustering(M, K):
	print "Finding normalized D"
	normD = np.zeros((6000, 6000))

	for i in range(M.shape[0]):
		normD[i,i] = (float)(1/math.sqrt(sum(M[i,])))
	print "Done finding normalized D"

	print "Finding normalized L"
	normL = np.subtract(np.identity(6000), np.dot(np.dot(normD, M), normD))
	print "Done finding normalized L"

	print "Finding eigenvalues and eigenvectors"
	evalues, vr = LA.eig(normL)
	print "Done finding eigenvalues and eigenvectors"
	
	print "Finding the smallest K eigenvectors"
	np_vectors = np.array(vr[:K])
	np_vectors = np_vectors.T
	print "Done finding the smallest K eigenvectors"

	print "Performing K-Means Clustering"
	codebook, _ = kmeans(np_vectors, 10)
	code, _ = vq(np_vectors, codebook)
	return code
	# clusters = [[],[],[],[],[],[],[],[],[],[]]
	# for i in range(code.shape[0]):
	# 	clusters[code[i]].append()
	# print "Done performing K-Means Clustering"


def get_labels_dict(seedFile):
	print "Filling in dictionary..."
	labels = {}
	for line in seedFile:
		(node, label) = [int(x) for x in line.split(',')]
		labels.setdefault(label, set()).add(node)
	print "Done filling in dictionary"
	return labels

def run_spectral_clustering(M):
	delta = np.std(M)
	M = np.exp(- M ** 2 / (2. * delta ** 2))

	spectral = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='precomputed')
	spectralClusters = spectral.fit_predict(M) # 1 x 6000
	
	clusters = {}
	for idx, cluster in enumerate(spectralClusters):
		clusters.setdefault(cluster, set()).add(idx+1)

	return clusters

def write_clusters_to_csv(clusters):
	with open('clusters.csv', 'wb') as f:
		for cluster, nodes in clusters.items():
			line = ','.join([str(n) for n in list(nodes)])
			f.write(line)
			f.write('\n')
	print "Wrote clusters to csv"

def load_clusters():
	with open('clusters.csv', 'r') as f:
		lines = f.readlines()

		clusters = {}
		for idx, line in enumerate(lines):
			clusters[idx] = set([int(e) for e in line.split(',')])

		return clusters

def load_adjacency_matrix():
	M = np.zeros((6000,6000))
	with open('adjacency_matrix.csv', 'r') as f:
		lines = f.readlines()
		for idx, line in enumerate(lines):
			M[idx,:] = [float(e) for e in line.split(',')]

	return M

def write_adjacency_matrix_to_csv(M):
	# Write matrix to CSV
	with open('adjacency_matrix.csv', 'wb') as f:
		for i in range(M.shape[0]):
			line = ','.join([str(e) for e in M[i,:]])
			f.write(line)
			f.write('\n')

# Run the program
def run():
	np.set_printoptions(threshold=np.nan)
	print "Creating features matrix"
	features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')
	print "Done creating features matrix"

	# graphFile = open('../Graph.csv', 'r')
	# M = findWeightedMatrix(graphFile, features) # Weighted (distance) adjacency matrix
	# write_adjacency_matrix_to_csv(M)

	# Load weighted adjacency matrix from file
	# M = load_adjacency_matrix()

	# # # Run spectral clustering to get 10 clusters
	# clusters = run_spectral_clustering(M)
	# write_clusters_to_csv(clusters)

	clusters = load_clusters()

	# Load seed labels
	seedFile = open('../Seed.csv', 'r')

	centroids = np.zeros((10,1084))
	for cluster, nodes in clusters.items():
		featureVectors = np.zeros((len(nodes), 1084))
		for i, node in enumerate(nodes):
			featureVectors[i,:] = features[node-1,:]

		avgVector = np.mean(featureVectors, axis=0)
		
		centroids[cluster,:] = avgVector
		
	# Run k-means on last 4000 nodes
	kmeans = KMeans(n_clusters=10, init=centroids, n_init=1)
	newClusters = kmeans.fit_predict(features)

	clusterAssignment = {}
	for node, cluster in enumerate(newClusters):
		clusterAssignment.setdefault(cluster, set()).add(node+1)

	labelledClusters = {}
	labelsDict = get_labels_dict(seedFile)
	for label, val in labelsDict.items():
		for node in val:
			for clusterIdx, nodes in clusterAssignment.items():
				if node in nodes:
					labelledClusters.setdefault(label, []).append(clusterIdx)

	print "Clusters dict"
	for num, clusters in labelledClusters.items():
		print num, ': ', clusters 

	clusterToDigit = {
		0: 8,
		1: 1,
		2: 2,
		3: 9, 
		4: 4,
		5: 7,
		6: 5,
		7: 6,
		8: 3,
		9: 0
	}

	output = np.zeros((10000,2), dtype='int32')
	for clusterIdx, nodes in clusterAssignment.items():
		for node in nodes:
			output[node-1,0] = node
			output[node-1,1] = clusterToDigit[clusterIdx]

	output = output[6000:,:]

	np.savetxt("spectral_kmeans.csv", output.astype(int), fmt='%i', delimiter=",", header="Id,Label", comments='')


if __name__ == '__main__':	
	run()
