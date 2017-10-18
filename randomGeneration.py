import numpy as np
from random import randint

def randomSubmission():
	submission = np.zeros((4000, 2), dtype='int32')
	for i in range(submission.shape[0]):
		submission[i, 0] = i+6001
		submission[i, 1] = 3

	np.savetxt("submission.csv", submission, fmt='%i', delimiter=",", header="Id,Label", comments='')

randomSubmission()