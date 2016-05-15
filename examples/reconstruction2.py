from __future__ import print_function

import sys

import numpy as np
from scipy import optimize

from utils.testEnv import TestEnv
from vision.features import generate_pairs

def makeReconstruction(path, sift_mat_path=None):
	myenv = TestEnv()
	myenv.copyAllResizedImages(path, y=1024, v=True)

	generate_pairs(myenv.imagesList, myenv.pairsPath)

	# myenv.close(v=True)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Give a path man..')
		exit() 

	if len(sys.argv) > 2:
		makeReconstruction(sys.argv[1], sys.argv[2])
	else:
		makeReconstruction(sys.argv[1])
	

