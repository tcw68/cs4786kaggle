import numpy as np
from scipy import linalg as LA
from collections import deque
import math

def unique_rows(M):
	a = np.ascontiguousarray(M)
	unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
	return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def findD():
	return


def findL():
	return


def runKMeans():
	return


## Returns 6000x6000 adjacency matrix
def createAdjacencyMatrix(graphFile):
	print "Creating adjacency matrix..."
	N = 6000
	matrix = [[0 for i in range(N)] for j in range(N)]

	print "Filling in matrix..."
	for line in graphFile:
		(node1, node2) = [int(x) for x in line.split(',')]
		matrix[node1-1][node2-1] = 1
		matrix[node2-1][node1-1] = 1

	graphFile.close()
	print "Done creating matrix"
	return matrix

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


# Returns the list of nodes most similar to a specific number
def similaritySearch(num, seedNodes, M):
	print "Similarity search for number", num, "..."
	similarNodes = []
	similarityRatio = 0.65

	while similarityRatio <= 1.0:
		for node in range(len(M)):
			numSimilar = 0
			for seed in seedNodes:
				if node == seed:
					continue
				if M[node][seed] == 1:
					numSimilar += 1
			if (numSimilar/float(len(seedNodes))) >= similarityRatio:
				similarNodes.append(node)
		seedNodes += similarNodes
		similarityRatio += 0.1

	return similarNodes

def spectralClustering(M, K):
	normD = np.zeros((6000, 6000))

	for i in range(M.shape[0]):
		normD[i,i] = (float)(1/math.sqrt(sum(M[i,])))

	normL = np.subtract(np.identity(6000), np.dot(np.dot(normD, M), normD)
	evalues, _, vr = LA.eig(normL)

	order = np.argsort(evalues)
	for i in range(K):
		



# Run the program
def run():
	print "Loading files..."
	featuresVectorFile = open('../Extracted_features.csv', 'r')
	seedFile = open('../Seed.csv', 'r')
	graphFile = open('../Graph.csv', 'r')
	print 'Done loading files'

	M = createAdjacencyMatrix(graphFile)
	Mnp = numpyMatrix(graphFile)

	spectralClustering(Mnp)

	# Initialize numbered lists with key=num and value=[node]
	numberedNodes = {}
	for i in range(10):
		numberedNodes[i] = []

	# Add seed nodes to numbered lists
	for line in seedFile:
		(node, num) = [int(x) for x in line.split(',')]
		numberedNodes[num].append(node)

	totalNodesLabeled = 0
	for num in numberedNodes:
		similarNodes = similaritySearch(num, numberedNodes[num], M)
		print "Length of nodes:", len(similarNodes)
		totalNodesLabeled += len(similarNodes)
		numberedNodes[num] = similarNodes
	print "Total nodes labeled:",totalNodesLabeled


run()
