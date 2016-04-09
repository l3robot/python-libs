import cv2
import numpy as np

from clint.textui import progress


"""
drawKeypoints
----------
draw points on an image
"""
def drawKeypoints(img, points):

	for p in points:
		a, b = p.ravel()
		img = cv2.circle(img,(a,b),5,(0,0,255),-1)

	cv2.imshow('frame',img)
	cv2.waitKey()


"""
siftExtraction
----------
Extract sift keypoints for a list of images
"""
def siftExtraction(imagesList, v=False):

	sift = cv2.xfeatures2d.SIFT_create()

	xdes = []

	if v == True:
		with progress.Bar(label=" [x] Sift extraction ...", expected_size=len(imagesList)) as bar:
			val = 0
			for i in imagesList:
				img = cv2.imread(i)
				gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				_, des = sift.detectAndCompute(gray,None)
				xdes.append(des)
				bar.show(val)
				val += 1
	else:
		for i in imagesList:
			img = cv2.imread(i)
			gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			_, des = sift.detectAndCompute(gray,None)
			xdes.append(des)

	return np.array(xdes)


"""
shiTomasiExtraction
----------
Extract Shi-Tomasi keypoints for a list of images
"""
def shiTomasiExtraction(imagesList, v=False):

	feature_params = dict( maxCorners = 100,
						   qualityLevel = 0.3,
						   minDistance = 7,
						   blockSize = 7 )

	lk_params = dict( winSize  = (15,15),
					  maxLevel = 2,
					  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	xdes = []

	old_image = cv2.imread(imagesList[0])
	old_gray = cv2.cvtColor(old_image,cv2.COLOR_BGR2GRAY)

	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	if v == True:
		with progress.Bar(label=" [x] Shi-Tomasi extraction ...", expected_size=len(imagesList)-1) as bar:
			val = 0
			for i in imagesList[1:]:
				new_image = cv2.imread(i)
				new_gray = cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)

				p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
				
				if len(p0) == 0:
					return np.array(xdes)
					
				xdes.append(p0[st==1])

				old_gray = new_gray.copy()
				p0 = p1[st==1].reshape(-1,1,2)

				bar.show(val)
				val += 1

			xdes.append(p1[st==1])
	else:
		for i in imagesList[1:]:
			new_image = cv2.imread(i)
			new_gray = cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)

			p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
			
			if len(p0) == 0:
				return np.array(xdes)
				
			xdes.append(p0[st==1])

			old_gray = new_gray.copy()
			p0 = p1[st==1].reshape(-1,1,2)

			bar.show(val)
			val += 1

		xdes.append(p1[st==1])

	return np.array(xdes)


"""
matchSiftKeypoints
----------
Match sift keypoints from a list of descriptors
"""
def matchSiftKeypoints(des, v=False):
	pass