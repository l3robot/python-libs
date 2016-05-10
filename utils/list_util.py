import numpy as np

"""
sort_together:
--------------
sorting x and y together
"""
def sort_together(x, y):
	idx = np.argsort(x)
	nx = np.sort(x)
	ny = [y[i] for i in idx]
	return nx, ny