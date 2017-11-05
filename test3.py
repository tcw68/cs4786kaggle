import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
import matplotlib.pyplot as plt
# import networkx as nx
from sklearn import neighbors
from scipy.sparse import csgraph
from itertools import combinations, permutations
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn import datasets
from sklearn import cross_decomposition as cd
from sklearn.semi_supervised import label_propagation
from sklearn import svm
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from time import time
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
import random
from scipy.stats import mode

import lda
import lda.datasets

print "Generating features matrix"
features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')
print "Done generating features matrix"

print "Generating similarity matrix"
M = np.genfromtxt("../Graph.csv", delimiter = ',')
print "Done generating similarity matrix"

print "Generating seed matrix"
seed = np.genfromtxt("../Seed.csv", delimiter = ',')
print "Done generating similarity matrix"


# Create distance matrix using updated matrix
# 0 = identical, larger number = greater dissimilarity, infinity = not similar at all
# def getWeightedAdjacencyMatrix(M):
# 	print "Creating features matrix..."
# 	features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')
# 	print "Done creating features matrix"

# 	print "Creating weighted adjacency matrix..."
# 	WM = np.full((6000,6000), float(100))
# 	for i in range(M.shape[0]):
# 		for j in range(M.shape[1]):
# 			if i == j: 
# 				WM[i,i] = 0
# 				continue
# 			if M[i,j] == 0: 
# 				continue
# 			elif M[i,j] == 1:
# 				euclidean_distance = np.linalg.norm(features[i] - features[j])
# 				WM[i][j] = euclidean_distance
# 				WM[j][i] = euclidean_distance
# 			elif M[i,j] == 2:
# 				WM[i][j] = 0
# 				WM[j][i] = 0

# 	print "Done creating weighted adjacency matrix"
# 	return WM

# def getWeightedAdjacencyMatrix(M):
# 	print "Creating features matrix..."
# 	features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')
# 	print "Done creating features matrix"

# 	print "Creating weighted adjacency matrix..."
# 	WM = np.zeros((6000,6000))
# 	for i in range(M.shape[0]):
# 		for j in range(M.shape[1]):
# 			if i == j: 
# 				WM[i,i] = 1
# 				continue
# 			if M[i,j] == 0: 
# 				continue
# 			elif M[i,j] == 1:
# 				euclidean_distance = np.linalg.norm(features[i] - features[j])
# 				WM[i][j] = euclidean_distance
# 				WM[j][i] = euclidean_distance
# 			elif M[i,j] == 2:
# 				WM[i][j] = 1
# 				WM[j][i] = 1

# 	print "Done creating weighted adjacency matrix"
# 	return WM

# def findWeightedMatrix(M):
# 	print "Creating features matrix..."
# 	features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')
# 	print "Done creating features matrix"

# 	print "Creating numpy adjacency matrix..."
# 	WM = np.full((6000,6000), float(100))

# 	print "Filling in matrix..."
# 	for i in range(M.shape[0]):
# 		for j in range(M.shape[1]):
# 			if i == j:
# 				WM[i,j] = 0
# 			if M[i,j] == 0: 
# 				continue
# 			else:
# 				euclidean_distance = np.linalg.norm(features[i] - features[j])
# 				WM[i,j] = euclidean_distance
# 				WM[j,i] = euclidean_distance

# 	print "Done creating matrix"
# 	return WM

def getWeightedMatrix(M):
	print "Creating features matrix..."
	features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')
	print "Done creating features matrix"

	print "Creating numpy adjacency matrix..."
	WM = np.zeros((6000,6000))

	print "Filling in matrix..."
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			if i == j:
				WM[i,j] = 1
			if M[i,j] == 0: 
				continue
			else:
				euclidean_distance = np.linalg.norm(features[i] - features[j])
				WM[i,j] = 1.0 / euclidean_distance
				WM[j,i] = 1.0 / euclidean_distance

	print "Done creating matrix"
	return WM

# Return unweighted adjacency matrix
# M[i][j] = 1 if nodes i+1 and j+1 are connected by an edge; else 0
def getUnweightedAdjacencyMatrix():
	with open('../Graph.csv', 'r') as f:
		print "Creating unweighted adjacency matrix..."

		M = np.zeros((6000,6000), dtype=np.int)
		lines = f.readlines()
		for line in lines:
			(node1, node2) = [int(x) for x in line.split(',')]
			M[node1-1][node2-1] = 1
			M[node2-1][node1-1] = 1

		print "Done creating unweighted adjacency matrix"
		return M

def writeUnweightedAdjacencyMatrixToCSV(fileName, M):
	with open(fileName, 'wb') as f:
		print "Writing unweighted adjacency matrix to CSV..."
		for row in M:
			line = ','.join([str(e) for e in row])
			f.write('%s\n' % line)
		print "Wrote unweighted adjacency matrix to CSV"

def loadUnweightedAdjacencyMatrix(fileName):
	with open(fileName, 'r') as f:
		print "Loading unweighted adjacency matrix..."

		M = np.zeros((6000,6000))
		lines = f.readlines()
		for i, line in enumerate(lines):
			M[i,:] = [int(e) for e in line.split(',')]

		print "Done loading unweighted adjacency matrix"
		return M.astype(int)

def loadWeightedAdjacencyMatrix(fileName):
	with open(fileName, 'r') as f:
		print "Loading unweighted adjacency matrix..."

		M = np.zeros((6000,6000))
		lines = f.readlines()
		for i, line in enumerate(lines):
			M[i,:] = [float(e) for e in line.split(',')]

		print "Done loading unweighted adjacency matrix"
		return M.astype(float)

def loadMatrix(fileName, row, col):
	with open(fileName, 'r') as f:
		print "Loading unweighted adjacency matrix..."

		M = np.zeros((row,col))
		lines = f.readlines()
		for i, line in enumerate(lines):
			M[i,:] = [float(e) for e in line.split(',')]

		print "Done loading unweighted adjacency matrix"
		return M.astype(float)

		
def getLabelsDict(): 
	with open('../Seed.csv', 'r') as f:
		print "Filling in correct labels dictionary..."

		labels = {}
		lines = f.readlines()
		for line in lines:
			(node, label) = [int(x) for x in line.split(',')]
			labels.setdefault(label, set()).add(node)

		print "Done filling in correct labels dictionary"
		return labels

def pprintDict(d):
	for key, val in d.items():
		print key, ':', val

def plotGraph(adjacency_matrix):
	print "Plotting graph for 6000 nodes..."
	rows, cols = np.where(adjacency_matrix == 1)
	edges = zip(rows.tolist(), cols.tolist())
	G = nx.Graph()
	G.add_edges_from(edges)
	nx.draw(G)
	plt.show()
	print "Done plotting graph for 6000 nodes"

# Print which nodes in correct labels aren't connected 
def printConnectedGraphs(M, labels):
	for digit, nodes in labels.items():
		print 'Digit: %i' % digit
		nodes = list(nodes)
		combos = list(combinations(nodes, 2))
		count = 0
		for (i, j) in combos:
			if M[i-1, j-1] == 0:
				print 'nodes %i and %i not connected' % (i, j)
			else:
				count += 1
		print 'Connected nodes: %i/%i' % (count, len(combos))
		print '\n'

# Update unweighted adjacency matrix using correctly labelled nodes
# 2 = definitely connected
# 1 = unknown connected
# 0 = not connected
def updateUnweightedAdjacencyMatrix(M, labels):
	labelledNodes = set() # Set of 60 correctly labelled nodes
	connectedNodes = set() # Set of (i,j) pairs where nodes i and j are connected
	for _, nodes in labels.items():
		labelledNodes.update(nodes)
		nodeCombos = set(combinations(nodes, 2))
		connectedNodes.update(nodeCombos)

	# All possible 2-element combos of 60 correctly labelled nodes
	allCombos = list(combinations(labelledNodes, 2))

	for (i, j) in allCombos:
		if (i, j) in connectedNodes or (j, i) in connectedNodes:
			M[i-1, j-1] = 1
			M[j-1, i-1] = 1
		else:
			M[i-1, j-1] = 0
			M[j-1, i-1] = 0

	return M

def runSpectralClustering(M):
	# delta = np.std(M)
	# M = np.exp(- M ** 2 / (2. * delta ** 2))
	features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')

	spectral = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='precomputed')
	spectralClusters = spectral.fit_predict(features) # 1 x 6000
	
	clusters = {}
	for idx, cluster in enumerate(spectralClusters):
		clusters.setdefault(cluster, set()).add(idx+1)

	return clusters

def writeClustersToCSV(clusters):
	with open('new_clusters.csv', 'wb') as f:
		for cluster, nodes in clusters.items():
			line = ','.join([str(n) for n in list(nodes)])
			f.write(line)
			f.write('\n')
	print "Wrote clusters to csv"

def validateClusters(clusterAssignments):
	labels = getLabelsDict()

	labelledClusters = {}
	for digit, labelledNodes in labels.items():
		for node in labelledNodes:
			for clusterIdx, nodes in clusterAssignments.items():
				if node in nodes:
					labelledClusters.setdefault(digit, []).append(clusterIdx)

	print "Clusters dict"
	for num, clusters in labelledClusters.items():
		print num, ': ', clusters 

def load_clusters():
	with open('clusters.csv', 'r') as f:
		lines = f.readlines()

		clusters = {}
		for idx, line in enumerate(lines):
			clusters[idx] = set([int(e) for e in line.split(',')])

		return clusters

def runLabelPropagationSVM():
	labels = getLabelsDict()

	# Dict of node to digit for 60 labelled nodes
	labelledNodes = {}
	for digit, nodes in labels.items():
		for node in nodes:
			labelledNodes[node] = digit

	features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')

	y_train = [-1] * 6000
	unlabeled_set = []
	for i in range(6000):
		if i+1 in labelledNodes:
			y_train[i] = labelledNodes[i+1]
		else:
			unlabeled_set.append(i)

	y_train = np.array(y_train)
	unlabeled_set = np.array(unlabeled_set)

	print "Start propagating labels..."
	lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
	lp_model.fit(features[:6000], y_train)
	predicted_labels = lp_model.transduction_[unlabeled_set]
	print "Done propagating labels"

	for idx, label in zip(unlabeled_set, predicted_labels):
		labels[label].add(idx+1)

	# for digit, nodes in labels.items():
	# 	print 'Digit %i: %i' % (digit, len(nodes))

	newLabelledNodes = {}
	for digit, nodes in labels.items():
		for node in nodes:
			newLabelledNodes[node] = digit

	y = []
	for i in range(len(newLabelledNodes)):
		y.append(newLabelledNodes[i+1])

	# for i in range(10):
	# 	print i, y.count(i)

	print "Start running SVM..."
	svmModel = svm.SVC()
	svmModel.fit(features[:6000], y)
	print "Finished running SVM"

	finalLabelsDict = {}
	final_labels = svmModel.predict(features[6000:])

	for idx, label in enumerate(final_labels):
		finalLabelsDict.setdefault(label, []).append(6001+idx)

	for digit, nodes in finalLabelsDict.items():
		print 'Digit %i: %i' % (digit, len(nodes))

	output = np.zeros((4000,2), dtype='int32')
	for digit, nodes in finalLabelsDict.items():
		for node in nodes:
			output[node-6001,0] = node
			output[node-6001,1] = digit

	np.savetxt("label_prop_svm.csv", output.astype(int), fmt='%i', delimiter=",", header="Id,Label", comments='')

def run_svm(train_set, test_set, train_labels, fileName):
	print "Start running SVM..."
	svmModel = svm.SVC()
	svmModel.fit(train_set, train_labels)
	print "Finished running SVM"

	finalLabelsDict = {}
	final_labels = svmModel.predict(test_set)

	for idx, label in enumerate(final_labels):
		finalLabelsDict.setdefault(label, []).append(6001+idx)

	for digit, nodes in finalLabelsDict.items():
		print 'Digit %i: %i' % (digit, len(nodes))

	output = np.zeros((4000,2), dtype='int32')
	for digit, nodes in finalLabelsDict.items():
		for node in nodes:
			output[node-6001,0] = node
			output[node-6001,1] = digit

	np.savetxt(fileName, output.astype(int), fmt='%i', delimiter=",", header="Id,Label", comments='')

def run_svm2(train_set, test_set, train_labels):
	print "Start running SVM..."
	svmModel = svm.SVC()
	svmModel.fit(train_set, train_labels)
	print "Finished running SVM"

	finalLabelsDict = {}
	final_labels = svmModel.predict(test_set)

	np.savetxt("labels_for_wrong.csv", final_labels.astype(int), fmt='%i', delimiter=",", header="Label", comments='')

# Do spectral clustering to get 10 clusters
def runSpectralClustering(M):
	# delta = np.std(M)
	# M = np.exp(- M ** 2 / (2. * delta ** 2))

	print "Performing spectral clustering..."
	spectral = SpectralClustering(n_clusters=10, affinity='precomputed', n_init=100, assign_labels='discretize')
	predicted_labels = spectral.fit_predict(M) # 1 x 6000
	print "Done performing spectral clustering"

	return predicted_labels

# Do agglomerative clustering to get 10 clusters
def runAgglomerativeClustering(X, linkage='ward'):
	print "Performing agglomerative clustering..."
	clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
	predicted_labels = clustering.fit_predict(X)
	print "Done performing agglomerative clustering"

	return predicted_labels

# Do K-means clustering to get 10 clusters
def runKMeansClustering(M):
	print "Performing KMeans clustering..."
	kmeans = KMeans(n_clusters=10)
	predicted_labels = kmeans.fit_predict(M)
	print "Done performing KMeans clustering"

	return predicted_labels

# Brute force 10! to find labelling that maximizes seed accuracy
def getOptimalClusterLabelling(clusterAssignments):
	print "Looking for optimal cluster labelling..."
	clusters = clusterAssignments.keys()
	digits = [i for i in range(10)]
	perms = list(permutations(digits)) # 10! possibilities

	labels = getLabelsDict()

	maxAccuracy = 0.0
	maxPerm = []
	for i, perm in enumerate(perms):
		numCorrectLabels = 0.0
		mapping = zip(clusters, perm) # cluster_no, digit
		for (cluster_no, digit) in mapping:
			clusterNodes = clusterAssignments[cluster_no]
			labelledNodes = labels[digit]
			overlap = clusterNodes.intersection(labelledNodes)
			numCorrectLabels += len(overlap)
		accuracy = numCorrectLabels / float(60)
		if accuracy > maxAccuracy:
			maxAccuracy = accuracy
			maxPerm = perm
	print "Found optimal cluster labelling"
	print "Max accuracy: %f" % maxAccuracy

	print "Optimal digit-cluster mapping"
	clusterMapping = zip(maxPerm, clusters)
	sortedMapping = sorted(clusterMapping, key=lambda x: x[0])
	for (digit, cluster) in sortedMapping:
		print digit, ':', cluster

	clusterDigitMapping = dict(zip(clusters, maxPerm))

	return clusterDigitMapping

def getClusterAssignments(predictedLabels):
	# labelledDict stores the cluster -> nodes in cluster
	labelledDict = {}
	for idx, label in enumerate(predictedLabels):
		node = idx + 1
		labelledDict.setdefault(label, set()).add(node)

	# Print out number of nodes in each cluster
	for cluster, nodes in labelledDict.items():
		print 'Cluster %i: %i' % (cluster, len(nodes))

	return labelledDict

#Find the nodes that are right and wrong and try running SVM again to fix the wrong nodes
def runModelTwice():
	bestScoreUsingGMM = np.genfromtxt('se_gmm_svm 3.csv', delimiter=',', dtype='int32', skip_header=1)
	bestScoreUsingKMeansXY = np.genfromtxt('se_cca80_kmeans_2000init_svm.csv', delimiter=',', dtype='int32', skip_header=1)
	gmmPlusModifying = np.genfromtxt('se_gmm_svm_modifying.csv', delimiter = ',', dtype='int32', skip_header=1)
	kMeansX = np.genfromtxt('se_cca_kmeans_svm.csv', delimiter=',', dtype='int32', skip_header=1)

	wrong = {}
	right = {}
	for j in range(thomas.shape[0]):
		if not (thomas[j, 1] == logan[j, 1] and thomas[j, 1] == modifying[j, 1] and thomas[j, 1] == k[j, 1]):
			wrong[thomas[j,0]] = (thomas[j, 1], logan[j, 1], modifying[j,1], k[j,1])
		else:
			right[thomas[j,0]] = thomas[j,1]

	npRight = np.zeros((3171, 1084))
	for i, k in enumerate(right.keys()):
		npRight[i,:] = features[k-1,:]

	npDigits = np.zeros((3171,1))
	for i,(k, v) in enumerate(right.items()):
		npDigits[i,:] = right[k]

	npTest = np.zeros((829,1084))
	for i,(k, v) in enumerate(wrong.items()):
		npTest[i,:] = features[k-1,:]

	training_set = np.concatenate((features[:6000,:], npRight), axis=0)
	labelsFor6000 = np.genfromtxt('labelsfor6000.csv', dtype='int32')
	labelsFor6000 = np.reshape(labelsFor6000, (6000, 1))
	training_labels = np.concatenate((labels6000, npDigits), axis=0)

	run_svm(training_set, npTest, training_labels, "newDigitsForWrong.csv")

	labelsWrong = np.genfromtxt('newDigitsForWrong.csv', dtype='int32')
	newDict = {}

	for i in range(replacement.shape[0]):
		bestScoreUsingGMM[i, 0] = labelsWrong[i, 1]

	np.savetxt('svmTwice.csv', bestScoreUsingGMM.astype(int), fmt='%i', delimiter=",", header="Id,Label", comments='')

	#Look at the final digits for those that were wrong and see if SVM
	#changed the digit to something that is unlikely such as 4 -> 0
	#because they don't look alike
	for k in wrong.iterkeys():
		print (bestScoreUsingKMeansXY[k-1, 1], bestScoreUsingGMM[k-1, 1])

#GMM results in very few or no 9s therefore replace the nodes that are 9s from
#KMeans to 9s in GMM
def replaceGMMwithKMeans9s(fileNameGMM, fileNameKMeans):
	gmm = np.genfromtxt(fileNameGMM, dtype='int32', delimiter = ',')
	kmeans = np.genfromtxt(fileNameKMeans. dtype='int32', delimiter = ',')

	for i in range(kmeans.shape[0]):
		if kmeans[i, 1] == 9:
			gmm[i, 1] = 9

	np.savetxt('gmm_9s_replaced.csv', gmm.astype(int), fmt='%i', delimiter=",", header="Id,Label", comments='')

def performCCA(k, training_set, target_set):
	print "Performing CCA"
	cca = cd.CCA(n_components = k)
	newX_se, newY_se = cca.fit_transform(training_set, target_set)
	np.savetxt('newX_se.csv', newX_se)
	np.savetxt('newY_se.csv', newY_se)
	print "Done performing CCA"

def loadCCA(fileNameX, fileNameY = ""):
	print "Loading CCA"
	X = np.genfromtxt(fileNameX, delimiter = ',')
	Y = []
	if not (fileNameY == ""):
		Y = np.genfromtxt(fileNameY, delimiter = ',')
	print "Done Loading CCA"

	return X, Y


def spectralEmbedding(M):
	se = manifold.SpectralEmbedding(n_components = 1084)
	X_se = se.fit_transform(M)
	np.savetxt("X_se.csv")

def loadSpectralEmbedding(fileName):
	return np.genfromtxt(fileName, delimier = ',')

def run():
	"""
	TRY THIS:

	How we can get > 70%:
	- Use the 6000 x 6000 edges matrix of 0 and 1 values as the affinity matrix to pass into spectral embedding
		- Note: I'm not completely sure what the value of n_components is supposed to be but I think 1084 could work?
	- Perform Spectral embedding to get X_se
		- What spectral embedding does is that it allows you to apply your own clustering algorithm
		- Spectral clustering is just a combination of spectral embedding and k-means clustering
		- We prob want to use single linkage clustering or our own clustering in place of k-means 
		  since k-means doesn't do that well.
	- Pass X_se into the clustering algorithm with n_clusters = 10
		- I'm using agglomerative clustering right now, which is related to single linkage clustering
	- Get the predicted labels for the 6000 nodes using the clustering algorithm
	- Using predicted labels, group the clusters into labelledDict
	- Brute force 10! to find the labelling for the clusters that maximizes seed accuracy (60 labelled nodes)
	- Label clusters such that seed accuracy is maximized (similar to cluster mapping we did before)
	- Now you should have a 1 x 6000 array where every element at index i corresponds to the digit that node i+1 is assigned to
		- Call this array Y for example
	- Now you can call run_svm function above
		- run_svm(train_set, test_set, train_labels)
			- train_set = features[:6000]
			- test_set = features[6000:]
			- train_labels = Y from above
		- Make sure to rename the csv in run_svm if you want a new output file
	- You should be able to submit the csv made in run_svm to Kaggle directly
	"""



	# Load in 6000 x 6000 edges matrix of 0 and 1
	# M = loadWeightedAdjacencyMatrix('newWeightedAdjacencyMatrix.csv')
	# M = loadUnweightedAdjacencyMatrix('updatedWeightedMatrx.csv')

	# replace72 = np.genfromtxt('replacescore72.csv')
	# replace75 = np.genfromtxt('replacescore75.csv')
	# replaceinit15 = np.genfromtxt('replaceinit15.csv')

	# replaceDict = {}
	# for i in replaceinit15:
	# 	if replaceDict.has_key(i):
	# 		replaceDict[i][0] = replaceDict[i][0]+1
	# 		replaceDict[i].append(15)
	# 	else:
	# 		replaceDict[i] = [1,15]
	# for i in replace72:
	# 	if replaceDict.has_key(i):
	# 		replaceDict[i][0] = replaceDict[i][0]+1
	# 		replaceDict[i].append(72)
	# 	else:
	# 		replaceDict[i] = [1, 72]
	# for i in replace75:
	# 	if replaceDict.has_key(i):
	# 		replaceDict[i][0] = replaceDict[i][0]+1
	# 		replaceDict[i].append(75)
	# 	else:
	# 		replaceDict[i] = [1, 75]
	# counter = []
	# for k, v in replaceDict.items():
	# 	if v[0] > 1 or 75 in v:
	# 		counter.append(k)
	# 	else:
	# 		if bool(random.getrandbits(1)):
	# 			counter.append(k)
	# print len(counter)

	# replace = np.array(counter)
	# np.savetxt('final_replace.csv', replace.astype(int))
	# quit()

	# init15_2 = np.genfromtxt('se_gmm_svm_init15_2.csv', delimiter = ',', skip_header=1, dtype='int32')
	# final_replace = np.genfromtxt('final_replace.csv', delimiter=',',dtype = 'int32')

	# for j in final_replace:
	# 	init15_2[j-6001,1] = 9

	# np.savetxt("se_gmm_svm_modifying.csv", init15_2.astype(int), fmt='%i', delimiter=",", header="Id,Label", comments='')
	# quit()


	# predictedLabels = sModel.predict(npTest)

	# clusterAssignments = getClusterAssignments(predictedLabels)
	# clusterDigitMapping = getOptimalClusterLabelling(clusterAssignments)
	# validateClusters(clusterAssignments)

	# # # M = getUnweightedAdjacencyMatrix()

	spectralEmbedding()


	new_se = np.concatenate((newX_se70, newY_se70), axis=1)

	print "Performing gmm"
	gmm7 = GaussianMixture(n_components=10)
	gmm8 = GaussianMixture(n_components=10)
	gmm9 = GaussianMixture(n_components=10)
	gmm10 = GaussianMixture(n_components=10)
	gmm11 = GaussianMixture(n_components=10)
	gmm12 = GaussianMixture(n_components=10)

	gmm7.fit(new_se)
	predictedLabels7 = gmm7.predict(new_se)

	clusterAssignments7 = getClusterAssignments(predictedLabels7)
	clusterDigitMapping7 = getOptimalClusterLabelling(clusterAssignments7)
	validateClusters(clusterAssignments7)

	train_labels7 = []
	for cluster in predictedLabels7:
		train_labels7.append(clusterDigitMapping7[cluster])

	train_labels7 = np.array(train_labels7)
	train_labels7 = np.reshape(train_labels7, (1,6000))

	gmm8.fit(new_se)
	predictedLabels8 = gmm8.predict(new_se)

	clusterAssignments8 = getClusterAssignments(predictedLabels8)
	clusterDigitMapping8 = getOptimalClusterLabelling(clusterAssignments8)
	validateClusters(clusterAssignments8)

	train_labels8 = []
	for cluster in predictedLabels8:
		train_labels8.append(clusterDigitMapping8[cluster])

	train_labels8 = np.array(train_labels8)
	train_labels8 = np.reshape(train_labels8, (1,6000))

	gmm9.fit(new_se)
	predictedLabels9 = gmm9.predict(new_se)

	clusterAssignments9 = getClusterAssignments(predictedLabels9)
	clusterDigitMapping9 = getOptimalClusterLabelling(clusterAssignments9)
	validateClusters(clusterAssignments9)

	train_labels9 = []
	for cluster in predictedLabels9:
		train_labels9.append(clusterDigitMapping9[cluster])

	train_labels9 = np.array(train_labels9)
	train_labels9 = np.reshape(train_labels9, (1,6000))

	gmm10.fit(new_se)
	predictedLabels10 = gmm10.predict(new_se)

	clusterAssignments10 = getClusterAssignments(predictedLabels10)
	clusterDigitMapping10 = getOptimalClusterLabelling(clusterAssignments10)
	validateClusters(clusterAssignments10)

	train_labels10 = []
	for cluster in predictedLabels10:
		train_labels10.append(clusterDigitMapping10[cluster])

	train_labels10 = np.array(train_labels10)
	train_labels10 = np.reshape(train_labels10, (1,6000))

	gmm11.fit(new_se)
	predictedLabels11 = gmm11.predict(new_se)

	clusterAssignments11 = getClusterAssignments(predictedLabels11)
	clusterDigitMapping11 = getOptimalClusterLabelling(clusterAssignments11)
	validateClusters(clusterAssignments11)

	train_labels11 = []
	for cluster in predictedLabels11:
		train_labels11.append(clusterDigitMapping11[cluster])

	train_labels11 = np.array(train_labels11)
	train_labels11 = np.reshape(train_labels11, (1,6000))

	gmm12.fit(new_se)
	predictedLabels12 = gmm12.predict(new_se)

	clusterAssignments12 = getClusterAssignments(predictedLabels12)
	clusterDigitMapping12 = getOptimalClusterLabelling(clusterAssignments12)
	validateClusters(clusterAssignments12)

	train_labels12 = []
	for cluster in predictedLabels12:
		train_labels12.append(clusterDigitMapping12[cluster])

	train_labels12 = np.array(train_labels12)
	train_labels12 = np.reshape(train_labels12, (1,6000))
	

	# #print "Performing Birch"
	# # brc = Birch(n_clusters=10)
	# # predictedLabels = brc.fit_predict(newX_se70)
	# #print "Done performing birch"

	print "Performing KMeans"
	predictedLabels = runKMeansClustering(new_se)
	predictedLabels2 = runKMeansClustering(new_se)
	predictedLabels3 = runKMeansClustering(new_se)
	predictedLabels4 = runKMeansClustering(new_se)
	predictedLabels5 = runKMeansClustering(new_se)
	predictedLabels6 = runKMeansClustering(new_se)

	# # print "Performing Agglomerative"
	# # A = kneighbors_graph(newX_se70, 800)
	# # linkage = AgglomerativeClustering(n_clusters = 10, linkage = 'ward', affinity='euclidean', connectivity=A)
	# # # linkage2 = AgglomerativeClustering(n_clusters = 10)
	# # predictedLabels = linkage.fit_predict(newX_se70)
	# # predictedLabels2 = linkage2.fit_predict(newY_se)
	# # print "Done performing Agglomerative"

	# # print "Performing Spectral Clustering"
	# # spectral = SpectralClustering(n_clusters = 10)
	# # # spectral2 = SpectralClustering(n_clusters = 10)
	# # predictedLabels = spectral.fit_predict(newX_se70)
	# # # predictedLabels2 = spectral2.fit_predict(ewY_se)
	# # print "Done performing spectral Clustering"

	# # # # Do clustering to get 10 clusters
	# # predictedLabels = runAgglomerativeClustering(X_se)
	# # predictedLabels = runSpectralClustering(M) # 1 x 6000ssssssss

	clusterAssignments = getClusterAssignments(predictedLabels)
	clusterDigitMapping = getOptimalClusterLabelling(clusterAssignments)
	validateClusters(clusterAssignments)

	clusterAssignments2 = getClusterAssignments(predictedLabels2)
	clusterDigitMapping2 = getOptimalClusterLabelling(clusterAssignments2)
	validateClusters(clusterAssignments2)

	clusterAssignments3 = getClusterAssignments(predictedLabels3)
	clusterDigitMapping3 = getOptimalClusterLabelling(clusterAssignments3)
	validateClusters(clusterAssignments3)

	clusterAssignments4 = getClusterAssignments(predictedLabels4)
	clusterDigitMapping4 = getOptimalClusterLabelling(clusterAssignments4)
	validateClusters(clusterAssignments4)

	clusterAssignments5 = getClusterAssignments(predictedLabels5)
	clusterDigitMapping5 = getOptimalClusterLabelling(clusterAssignments5)
	validateClusters(clusterAssignments5)

	clusterAssignments6 = getClusterAssignments(predictedLabels6)
	clusterDigitMapping6 = getOptimalClusterLabelling(clusterAssignments6)
	validateClusters(clusterAssignments6)

	# clusterAssignments2 = getClusterAssignments(predictedLabels2)
	# clusterDigitMapping2 = getOptimalClusterLabelling(clusterAssignments2)
	# validateClusters(clusterAssignments2)

	train_labels = []
	for cluster in predictedLabels:
		train_labels.append(clusterDigitMapping[cluster])

	train_labels = np.array(train_labels)
	train_labels = np.reshape(train_labels, (1,6000))

	train_labels2 = []
	for cluster in predictedLabels2:
		train_labels2.append(clusterDigitMapping2[cluster])

	train_labels2 = np.array(train_labels2)
	train_labels2 = np.reshape(train_labels2, (1,6000))


	train_labels3 = []
	for cluster in predictedLabels3:
		train_labels3.append(clusterDigitMapping3[cluster])

	train_labels3 = np.array(train_labels3)
	train_labels3 = np.reshape(train_labels3, (1,6000))

	train_labels4 = []
	for cluster in predictedLabels4:
		train_labels4.append(clusterDigitMapping4[cluster])

	train_labels4 = np.array(train_labels4)
	train_labels4 = np.reshape(train_labels4, (1,6000))

	train_labels5 = []
	for cluster in predictedLabels5:
		train_labels5.append(clusterDigitMapping5[cluster])

	train_labels5 = np.array(train_labels5)
	train_labels5 = np.reshape(train_labels5, (1,6000))

	train_labels6 = []
	for cluster in predictedLabels6:
		train_labels6.append(clusterDigitMapping6[cluster])

	train_labels6 = np.array(train_labels6)
	train_labels6 = np.reshape(train_labels6, (1,6000))



	final_labels = np.concatenate((train_labels, train_labels2, train_labels3, train_labels4, train_labels5, train_labels6, train_labels7, train_labels8, train_labels9, train_labels10, train_labels11, train_labels12), axis=0)
	np.savetxt('concatenated_gmm_kmeans.csv', final_labels.astype(int), fmt='%i', delimiter=",", comments='')
	# final_labels = np.genfromtxt('concatenated_kmeans.csv', delimiter = ',', dtype='int32')
	train_labels, count = mode(final_labels)
	np.savetxt('train_labels2.csv', np.reshape(train_labels, (6000, )).astype(int), fmt='%i', delimiter=',', comments='')

	# train_labels2 = []
	# for cluster in predictedLabels:
	# 	train_labels2.append(clusterDigitMapping2[cluster])

	# train_labels2 = np.array(train_labels2)

	# # # Run SVM
	train_set, test_set = features[:6000], features[6000:]
	run_svm(train_set, test_set, np.reshape(train_labels, (6000, )))
	# run_svm(train_set, test_set, train_labels2)


	"""
	IGNORE COMMENTS BELOW THIS (WAS TESTING OUT RANDOM STUFF)
	"""

	# clusterAssignments = load_clusters()
	# validateClusters(clusterAssignments)



	# M = loadUnweightedAdjacencyMatrix('unweightedAdjacencyMatrix.csv')	
	# WM = findWeightedMatrix(M)

	# 1 : [2, 5, 111, 50, 19, 37]

	# labels = getLabelsDict()

	# for label, nodes in labels.items():
	# 	print "label %i" % label
	# 	result = []
	# 	for i in range(6000):
	# 		count = 0
	# 		for node in nodes:
	# 			if M[i,node-1] == 1:
	# 				count += 1
	# 		if count == 6: result.append(i+1)
	# 	print len(result)

	# combos = list(combinations(labels[1],2))
	# for (i, j) in combos:
	# 	if M[i,j] == 0:
	# 		print "nodes %i and %i are not connected" % (i, j)

	# clusterAssignments = runSpectralClustering(WM)
	# validateClusters(clusterAssignments)

	# features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')

	# labels = getLabelsDict()


	# # seedFile = open('../Seed.csv', 'r')

	# # centroids = np.zeros((10,1084))
	# # for cluster, nodes in clusters.items():
	# # 	featureVectors = np.zeros((len(nodes), 1084))
	# # 	for i, node in enumerate(nodes):
	# # 		featureVectors[i,:] = features[node-1,:]

	# # 	avgVector = np.mean(featureVectors, axis=0)
		
	# # 	centroids[cluster,:] = avgVector

	# for digit, nodes in labels.items():
	# 	print "Digit %i" % digit
	# 	combos = list(combinations(nodes, 2))
	# 	totalDist = 0.0
	# 	for (node1, node2) in combos:
	# 		feature1, feature2 = features[node1-1], features[node2-1]
	# 		euclideanDist = np.linalg.norm(feature1 - feature2)
	# 		print "Euclidean Distance b/w %i and %i is %f" % (node1, node2, euclideanDist)
	# 		totalDist += euclideanDist
	# 	print "Avg Distance: %f" % (totalDist / len(combos))



	# for (node1, node2) in combos:
	# 	feature1, feature2 = features[node1-1], features[node2-1]
	# 	similarity = 1 - spatial.distance.cosine(feature1, feature2)
	# 	print "Similarity b/w %i and %i is %f" % (node1, node2, similarity)

	# print '\n'

	# for node1 in nodes:
	# 	for node2 in labels[5]:
	# 		feature1, feature2 = features[node1-1], features[node2-1]
	# 		similarity = 1 - spatial.distance.cosine(feature1, feature2)
	# 		print "Similarity b/w %i and %i is %f" % (node1, node2, similarity)





 
# You can see here that running spectral clustering on the adjacency matrix
# for the 60 labelled nodes gets the perfect clustering
def checkLabelledNodes():
 	labels = getLabelsDict()

	labelledNodes = set() # Set of 60 correctly labelled nodes
	connectedNodes = set() # Set of (i,j) pairs where nodes i and j are connected
	for _, nodes in labels.items():
		labelledNodes.update(nodes)
		nodeCombos = set(combinations(nodes, 2))
		connectedNodes.update(nodeCombos)

	idxToNodeMapping = {}
	nodeToIdxMapping = {}
	for idx, node in enumerate(list(labelledNodes)):
		nodeToIdxMapping[node] = idx
		idxToNodeMapping[idx] = node

	M = np.zeros((60,60), dtype=np.int)
	for (node1, node2) in connectedNodes:
		idx1, idx2 = nodeToIdxMapping[node1], nodeToIdxMapping[node2]
		M[idx1, idx2] = 1
		M[idx2, idx1] = 1

	clusterAssignments = runSpectralClustering(M)

	for clusterIdx, nodes in clusterAssignments.items():
		clusterAssignments[clusterIdx] = [idxToNodeMapping[n-1] for n in nodes]

	labelledClusters = {}
	for digit, labelledNodes in labels.items():
		for node in labelledNodes:
			for clusterIdx, nodes in clusterAssignments.items():
				if node in nodes:
					labelledClusters.setdefault(digit, []).append(clusterIdx)

	print "Digit | Clusters"
	for num, clusters in labelledClusters.items():
		print num, '  :  ', clusters 


if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	run()
