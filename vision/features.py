from __future__ import print_function

import cv2
import numpy as np

import os

import copy

from clint.textui import progress


"""
drawKeypoints
----------
draw points on an image
"""


def drawKeypoints(img, points):

	for p in points:
		a, b = p.ravel()
		img = cv2.circle(img, (a, b), 5, (0, 0, 255), -1)

	cv2.imshow('frame', img)
	cv2.waitKey()


"""
siftExtraction
----------
Extract sift keypoints for a list of images
"""


def siftExtraction(imagesList, v=False):

	sift = cv2.xfeatures2d.SIFT_create()

	xdes = []
	xkpt = []

	if v == True:
		with progress.Bar(label=" [x] Sift extraction ...", expected_size=len(imagesList)) as bar:
			val = 0
			for i in imagesList:
				img = cv2.imread(i)
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				kpt, des = sift.detectAndCompute(gray, None)
				xdes.append(des)
				xkpt.append([k.pt for k in kpt])
				val += 1
				bar.show(val)
	else:
		for i in imagesList:
			img = cv2.imread(i)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			kpt, des = sift.detectAndCompute(gray, None)
			xdes.append(des)
			xkpt.append([k.pt for k in kpt])

	return np.array(xdes), np.array(xkpt)


"""
shiTomasiExtraction
----------
Extract Shi-Tomasi keypoints for a list of images
"""


def shiTomasiExtraction(imagesList, v=False):

	feature_params = dict(maxCorners=100,
						   qualityLevel=0.3,
						   minDistance=7,
						   blockSize=7)

	lk_params = dict(winSize=(15, 15),
					  maxLevel=2,
					  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	xdes = []

	old_image = cv2.imread(imagesList[0])
	old_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)

	p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

	if v == True:
		with progress.Bar(label=" [x] Shi-Tomasi extraction ...", expected_size=len(imagesList) - 1) as bar:
			val = 0
			for i in imagesList[1:]:
				new_image = cv2.imread(i)
				new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

				p1, st, err = cv2.calcOpticalFlowPyrLK(
				    old_gray, new_gray, p0, None, **lk_params)

				if len(p0) == 0:
					return np.array(xdes)

				xdes.append(p0[st == 1])

				old_gray = copy.copy(new_gray)
				p0 = p1[st == 1].reshape(-1, 1, 2)

				bar.show(val)
				val += 1

			xdes.append(p1[st == 1])
	else:
		for i in imagesList[1:]:
			new_image = cv2.imread(i)
			new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

			p1, st, err = cv2.calcOpticalFlowPyrLK(
			    old_gray, new_gray, p0, None, **lk_params)

			if len(p0) == 0:
				return np.array(xdes)

			xdes.append(p0[st == 1])

			old_gray = copy.copy(new_gray)
			p0 = p1[st == 1].reshape(-1, 1, 2)

			bar.show(val)
			val += 1

		xdes.append(p1[st == 1])

	return np.array(xdes)


"""
matchSift
----------
Match sift keypoints from a list of descriptors
"""


def matchSift(matcher, des1, des2, v=False):
	matches = matcher.knnMatch(des1, des2, k=2)

	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append(np.array([m.queryIdx, m.trainIdx]))

	if v == True:
		print('    [o] Found {} matches'.format(len(good)))

	return good


def matchSift2(matcher, des1, des2, v=False):
	matches = matcher.knnMatch(des1, des2, k=2)

	agood = []
	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			agood.append(np.array([m.queryIdx, m.trainIdx]))
			good.append(m)

	if v == True:
		print('    [o] Found {} matches'.format(len(good)))

	return agood, good


"""
matchSiftDescriptors
----------
Match sift descriptors from a list of descriptors
"""


def matchSiftDescriptors(des, algo, v=False):

	if algo not in ['flann', 'brute']:
		print(' [!] {} is not a good matcher algo'.format(algo))
		raise ValueError

	if algo == 'flann':
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)

		mymatcher = cv2.FlannBasedMatcher(index_params, search_params)
	elif algo == 'brute':
		mymatcher = cv2.BFMatcher()

	n = len(des)

	xmatches = np.zeros((n, n), dtype=np.ndarray)

	totaln = n * (n + 1) / 2

	if v == True:
		with progress.Bar(label=" [x] Matching sift descriptors ...", expected_size=totaln) as bar:
			val = 0
			for i in range(n):
				for j in range(i, n):
					matches = matchSift(mymatcher, des[i], des[j])
					xmatches[i, j] = matches
					xmatches[j, i] = np.array([m[::-1] for m in matches])
					val += 1
					bar.show(val)
	else:
		for i in range(n):
			for j in range(i, n):
				matches = matchSift(mymatcher, des[i], des[j])
				xmatches[i, j] = matches
				xmatches[j, i] = np.array([m[::-1] for m in matches])

	return xmatches


def homographyFilter(kp1, kp2, match, v=False):

	if len(match) > 10:
		query_pts = np.float32([kp1[m[0]] for m in match]).reshape(-1, 1, 2)
		train_pts = np.float32([kp2[m[1]] for m in match]).reshape(-1, 1, 2)

		M, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

		if M is None:
			if v == True:
				print('    [o] Failed to find homography')
			return []

		matchesMask = mask.ravel().tolist()

		if v == True:
			m_len = len(match)
			n_len = np.sum(matchesMask)
			perc = n_len / m_len * 100.0
			print('    [o] {} of {} matches kept ({:.2f}%)'.format(n_len, m_len, perc))

		nmatch = [m for i, m in enumerate(match) if matchesMask[i] == 1]

		return nmatch

	else:
		if v == True:
			print('    [o] Not enough matches')
		return []


def homographyFilter2(kp1, kp2, match, v=False):

	xkp1 = [k.pt for k in kp1]
	xkp2 = [k.pt for k in kp2]

	if len(match) > 10:
		query_pts = np.float32([xkp1[m[0]] for m in match]).reshape(-1, 1, 2)
		train_pts = np.float32([xkp2[m[1]] for m in match]).reshape(-1, 1, 2)

		M, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

		if M is None:
			if v == True:
				print('    [o] Failed to find homography')
			return []

		matchesMask = mask.ravel().tolist()

		if v == True:
			m_len = len(match)
			n_len = np.sum(matchesMask)
			perc = n_len / m_len * 100.0
			print('    [o] {} of {} matches kept ({:.2f}%)'.format(n_len, m_len, perc))

		return matchesMask

	else:
		if v == True:
			print('    [o] Not enough matches')
		return []


def homographyFilterPairs(kps, matches, v=False):

	n, m = matches.shape

	if v == True:
		with progress.Bar(label=" [x] Filtering matches ...", expected_size=n * m) as bar:
			val = 0
			for i in range(n):
				for j in range(m):
					nmatch = homographyFilter(kps[i], kps[j], matches[i, j])
					matches[i, j] = nmatch
					val += 1
					bar.show(val)
	else:
		for i in range(n):
			for j in range(m):
				nmatch = homographyFilter(kps[i], kps[j], matches[i, j])
				matches[i, j] = nmatch


def generate_pairs(imagesList, out):

	# sift = cv2.xfeatures2d.SIFT_create()
    fast = cv2.FastFeatureDetector()

	xdes = []
	xkpt = []

	with progress.Bar(label=" [x] Sift extraction ...", expected_size=len(imagesList)) as bar:
		for val, i in enumerate(imagesList):
			img = cv2.imread(i)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# kpt, des = sift.detectAndCompute(gray,None)
            kpt = fast.detect(img, None)
            des = fast.compute(img, kpt)

			xdes.append(des)
			xkpt.append(kpt)

			bar.show(val)

	mymatcher = cv2.BFMatcher()

	n = len(des)

	totaln = n*(n+1)/2

	with progress.Bar(label=" [x] Matching sift descriptors ...", expected_size=totaln) as bar:
		val = 0
		for i in range(n):
			for j in range(i,n):
				img1 = cv2.imread(imagesList[i])
				img2 = cv2.imread(imagesList[j])
				kp1 = xkpt[i]
				kp2 = xkpt[j]
				agood, good = matchSift2(mymatcher, xdes[i], xdes[j])
				matchesMask = homographyFilter2(kp1, kp2, agood, v=False)
				if len(matchesMask) > 0: 
					draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
					img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
					cv2.imwrite(os.path.join(out, '{}-{}.jpg'.format(i,j)),img3)
				val += 1
				bar.show(val)
