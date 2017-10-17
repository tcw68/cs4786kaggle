import numpy as np
import scipy as sp
from collections import deque

def unique_rows(M):
	a = np.ascontiguousarray(M)
	unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
	return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

## Returns 6000x6000 adjacency matrix
def createAdjacencyMatrix(graphFile):
	matrix = np.zeros((6000, 6000), dtype='int32')

	for line in graphFile:
		(node1, node2) = line.split(',')
		matrix[node1-1, node2-1] = 1
		matrix[node2-1, node1-1] = 1

	graphFile.close()
	return matrix

def findD():
	return

def findL():
	return

def runKMeans():
	return

# Returns a list of nodes similar (through transitivity) known numbered-nodes
def findVisited(visited, M):
	visitNext = deque()
	for i in visited:
		visitNext.append(i)

	while len(visitNext) is not 0:
		element = visitNext.popLeft()
		print "element", element

		for j in range(M.shape[1]):
			if M[element-1, j] == 1 and j not in visited:
				visited.append(j)
				visitNext.append(j)

		print "length", len(visitNext)

	return visited

def run():
	print "Loading files..."
	featuresVectorFile = open('Extracted_features.csv', 'r');
	seedFile = open('Seed.csv', 'r');
	graphFile = open('Graph.csv', 'r');
	print 'Done loading files'

	M = createAdjacencyMatrix(graphFile);


	np.set_printoptions(threshold='nan')
	print "starting features..."
	featuresVector = np.genfromtxt('Extracted_features.csv', delimiter=',')
	print "done features"
	print "starting seeds..."
	seedPoints = np.genfromtxt('Seed.csv', delimiter=',', dtype='int32')
	print "end seeds"

	M = createAdjacencyMatrix()

	listofVisited = [[], [], [], [], [], [], [], [], [], []] # 0-9
	finalVisited = []

	for i in range(seedPoints.shape[0]):
		node = seedPoints[i, 0]
		digit = seedPoints[i, 1]

		listofVisited[digit].append(node)

	print "starting findVisited"

	for i in range(len(listofVisited)):
		v = findVisited(listofVisited[i], M)
		print v
		finalVisited.append(v)

run()
