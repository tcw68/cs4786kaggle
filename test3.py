import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations

from sklearn import svm, manifold
from sklearn import cross_decomposition as cd
from sklearn.semi_supervised import label_propagation
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

# Load given CSV files
print "Generating features matrix"
features = np.genfromtxt('../Extracted_features.csv', delimiter = ',')
print "Done generating features matrix"

print "Generating similarity matrix"
M = np.genfromtxt("../Graph.csv", delimiter = ',')
print "Done generating similarity matrix"

print "Generating seed matrix"
seed = np.genfromtxt("../Seed.csv", delimiter = ',')
print "Done generating similarity matrix"

# Train on first 6000 nodes and test on last 4000 nodes
train_set, test_set = features[:6000], features[6000:]

# Find the weighted adjacency matrix using the feature vectors
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

# Write the adjacency matrix to csv
def writeUnweightedAdjacencyMatrixToCSV(fileName, M):
	with open(fileName, 'wb') as f:
		print "Writing unweighted adjacency matrix to CSV..."
		for row in M:
			line = ','.join([str(e) for e in row])
			f.write('%s\n' % line)
		print "Wrote unweighted adjacency matrix to CSV"

# Load adjacency matrix from csv
def loadUnweightedAdjacencyMatrix(fileName):
	with open(fileName, 'r') as f:
		print "Loading unweighted adjacency matrix..."

		M = np.zeros((6000,6000))
		lines = f.readlines()
		for i, line in enumerate(lines):
			M[i,:] = [int(e) for e in line.split(',')]

		print "Done loading unweighted adjacency matrix"
		return M.astype(int)

# Load weighted adjacency matrix from csv
def loadWeightedAdjacencyMatrix(fileName):
	with open(fileName, 'r') as f:
		print "Loading unweighted adjacency matrix..."

		M = np.zeros((6000,6000))
		lines = f.readlines()
		for i, line in enumerate(lines):
			M[i,:] = [float(e) for e in line.split(',')]

		print "Done loading unweighted adjacency matrix"
		return M.astype(float)

# Load any matrix from csv, need to specify the number of rows and columns
def loadMatrix(fileName, row, col):
	with open(fileName, 'r') as f:
		print "Loading unweighted adjacency matrix..."

		M = np.zeros((row,col))
		lines = f.readlines()
		for i, line in enumerate(lines):
			M[i,:] = [float(e) for e in line.split(',')]

		print "Done loading unweighted adjacency matrix"
		return M.astype(float)

# Find which seed nodes are for each digit		
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

# Print the dictionary
def printDict(d):
	for key, val in d.items():
		print key, ':', val

# Plot the adjacency matrix
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

# Run spectral clustering and return the clusters
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

# Write the clusters returned form a clustering algorithm to CSV
def writeClustersToCSV(clusters):
	with open('new_clusters.csv', 'wb') as f:
		for cluster, nodes in clusters.items():
			line = ','.join([str(n) for n in list(nodes)])
			f.write(line)
			f.write('\n')
	print "Wrote clusters to csv"

# Print out which nodes are in which clusters
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

# Load clusters from CSV file
def load_clusters():
	with open('clusters.csv', 'r') as f:
		lines = f.readlines()

		clusters = {}
		for idx, line in enumerate(lines):
			clusters[idx] = set([int(e) for e in line.split(',')])

		return clusters

# Label propagation with SVM, did not result in a very good score
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

# Run svm on the training set and predict labels for the test set
# Save the predicted labels for the test set to fileName
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

# SVM ran again on the wrong nodes to try and better predict the labels
def run_svm_again(train_set, test_set, train_labels, fileName):
	print "Start running SVM..."
	svmModel = svm.SVC()
	svmModel.fit(train_set, train_labels)
	print "Finished running SVM"

	final_labels = svmModel.predict(test_set)

	np.savetxt(fileName, final_labels.astype(int), fmt='%i', delimiter=",", header="Label", comments='')

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

# Do Gaussian Mixture Model Clustering to get 10 clusters
def runGaussianMixture(M):
	print "Performing Gaussian Mixture"
	gmm = GaussianMixture(n_components = 10, n_init = 15)
	gmm.fit(M)
	predicted_labels = gmm.predict(M)
	print "Done performing Gaussian Mixture"

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

# Find how many nodes are in each cluster
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

# Find the nodes that are right and wrong and try running SVM again to fix the wrong nodes
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

	run_svm_again(training_set, npTest, training_labels, "newDigitsForWrong.csv")

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

# GMM results in very few or no 9s therefore replace the nodes that are 9s from
# KMeans to 9s in GMM
def replaceGMMwithKMeans9s(fileNameGMM, fileNameKMeans):
	gmm = np.genfromtxt(fileNameGMM, dtype='int32', delimiter = ',')
	kmeans = np.genfromtxt(fileNameKMeans, dtype='int32', delimiter = ',')

	for i in range(kmeans.shape[0]):
		if kmeans[i, 1] == 9:
			gmm[i, 1] = 9

	np.savetxt('gmm_9s_replaced.csv', gmm.astype(int), fmt='%i', delimiter=",", header="Id,Label", comments='')

# Perform CCA on two different views and reduce to the k most correlated dimensions
def performCCA(k, training_set, target_set, fileNameX, fileNameY):
	cca = cd.CCA(n_components = k)
	newX_se, newY_se = cca.fit_transform(training_set, target_set)
	np.savetxt(fileNameX, newX_se)
	np.savetxt(fileNameY, newY_se)

# Load CCA results from CSV files
def loadCCA(fileNameX, fileNameY = ""):
	X = np.genfromtxt(fileNameX, delimiter = ',')
	Y = []
	if not (fileNameY == ""):
		Y = np.genfromtxt(fileNameY, delimiter = ',')

	return X, Y

# Perform Spectral Embedding on the similarity graph to 1084
def spectralEmbedding(M, fileName):
	se = manifold.SpectralEmbedding(n_components = 1084)
	X_se = se.fit_transform(M)
	np.savetxt(fileName, X_se, delimiter=',')

# Load the results of Spectral Embedding from a CSV file
def loadSpectralEmbedding(fileName):
	return np.genfromtxt(fileName, delimier = ',')

# Find the digit labels for each node in the training set
def findTrainingLabels(labels):
	clusterAssignments = getClusterAssignments(labels)
	clusterDigitMapping = getOptimalClusterLabelling(clusterAssignments)
	validateClusters(clusterAssignments)

	train_labels = []
	for cluster in labels:
		train_labels.append(clusterDigitMapping[cluster])

	train_labels = np.array(gmm_train_labels)
	train_labels = np.reshape(gmm_train_labels, (6000, ))

	return train_labels
 
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

def run():
	print "Performing Spectral Embedding"
	spectralEmbedding(M, "spectral_embedding.csv")
	X_se = loadSpectralEmbedding("spectral_embedding.csv")
	print "Done performing Spectral Embedding"

	print "Performing CCA"
	performCCA(70, X_se, features[:6000, :], "newX.csv", "newY.csv")
	X, Y = loadCCA("newX.csv", "newY.csv")
	print "Done performing CCA"

	if not (Y == []):
		new_se = np.concatenate((X, Y), axis=1)

	#Predict clusters using GMM, use new_se if using both X and Y
	gmm_predicted_labels = runGaussianMixture(X)

	gmm_train_labels = findTrainingLabels(gmm_predicted_labels)

	run_svm(train_set, test_set, gmm_train_labels, "se_cca_gmm_svm.csv")

	#Predict clusters using KMeans, use X if using just X
	kmeans_predicted_labels = runKMeansClustering(new_se)

	kmeans_train_labels = findTrainingLabels(kmeans_predicted_labels)

	run_svm(train_set, test_set, kmeans_train_labels, "se_cca_kmeans_svm.csv")

	#Predict clusters using agglomerative, use new_se if using both X and Y
	agglomerative_predicted_labels = runAgglomerativeClustering(X)

	agglomerative_train_labels = findTrainingLabels(agglomerative_predicted_labels)

	run_svm(train_set, test_set, agglomerative_train_labels, "se_cca_agglomerative_svm.csv")

if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	run()
