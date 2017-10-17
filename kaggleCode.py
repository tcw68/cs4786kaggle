import numpy as np
import scipy as sp
from collections import deque

# def DFS(M, visited, current, ):

# 	for j in len(M[current]):
# 		if (M[current][j] == 1 && visited[j] = 0):
# 			visited[j] = 1
# 			compressSimilarity.append(current)
# 			DFS(M, visited, j)


def unique_rows(M):
	a = np.ascontiguousarray(M)
	unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
	return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def findSimilarityMatrix():
	print "starting graph.csv"
	similarityCSV = np.genfromtxt('Graph.csv', delimiter=',', dtype='int32')
	print "done graph.csv"
	similarityMatrix = np.zeros((6000, 6000), dtype='int32')
	# visited = np.zeros((similarityMatrix[0]))
	# compressSimilarity = []
	# final = []

	for i in range(similarityCSV.shape[0]):
		row = similarityCSV[i, 0]
		column = similarityCSV[i, 1]

		similarityMatrix[row-1, column-1] = 1
		similarityMatrix[column-1, row-1] = 1

	# for i in range(1, similarityMatrix.shape[0]):
	# 	for j in range(i, similarityMatrix.shape[0]):
	# 		for k in range(1, similarityMatrix.shape[1]):
	# 				if (i==j):
	# 					continue
	# 				if similarityMatrix[i,k] == 1 and similarityMatrix[j, k] == 1:
	# 					for l in range(1, similarityMatrix.shape[1]):
	# 						similarityMatrix[i, l] = similarityMatrix[i, l] | similarityMatrix[j, l]
	# 						similarityMatrix[j, l] = similarityMatrix[i, l] | similarityMatrix[j, l]

	# np.savetxt("similarity_matrix.csv", similarityMatrix, delimiter=",")
	return similarityMatrix

	# for i in len(visited):
	# 	if (visited[i]==0):
	# 		compressSimilarity = []
	# 		DFS(M,visited,i, compressSimilarity)

	# return visited

def findD():
	return

def findL():
	return

def runKMeans():
	return

def findVisited(visited, M):
	visitNext = deque()
	for i in visited:
		visitNext.appendleft(i)

	while len(visitNext) is not 0:
		element = visitNext.pop()
		print "element", element

		for j in range(M.shape[1]):
			if M[element-1, j] == 1 and j not in visited:
				visited.append(j)
				visitNext.appendleft(j)

		print "length", len(visitNext)

	return visited


def run():
	np.set_printoptions(threshold='nan')
	print "starting features"
	featuresVector = np.genfromtxt('Extracted_features.csv', delimiter=',')
	print "done features"
	print "starting seeds"
	seedPoints = np.genfromtxt('Seed.csv', delimiter=',', dtype='int32')
	print "end seeds"
	M = findSimilarityMatrix()
	# D = np.zeros_like(M)

	listofVisited = [[], [], [], [], [], [], [], [], [], []]
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


	# for i in range(M.shape[0]):
	# 	total = sum(M[i,])
	# 	D[i, i] = total

	# L = D - M
	# _, _, vr = sp.linalg.eig(L)

run()
# findSimilarityMatrix()