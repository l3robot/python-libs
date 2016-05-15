from __future__ import print_function

import os
import time

from shutil import move, copytree, rmtree
from shutil import copy as scopy

from wand.image import Image
import cv2

import copy

from clint.textui import progress

from utils.findImage import drawImages, selectImages

# Constants #####################

test_dir = "/dev/shm/"

#############################

class TestEnv:

	def __init__(self, name=None):
		self.closed = False

		test_time = time.asctime(time.localtime())
		test_time = test_time.lower().replace(" ", "_")

		if name is None:	
			self.name = os.path.join(test_dir, test_time)
		else:
			self.name = os.path.join(test_dir, name + '-' + test_time)

	def createEnv(self, v=False):
		if self.closed == True:
			print(' [!] This environment is close')
			return

		if v == True:
			print(" [x] Creating test environment in {}".format(self.name))

		if not os.path.exists(self.name):
			os.makedirs(self.name)
		else:
			self.name = self.name+"-v2"
			os.makedirs(self.name)

	def copyRandomImages(self, origin, number, v=False):
		if self.closed == True:
			print(' [!] This environment is close')
			return

		self.imagesPath = os.path.join(self.name, 'images')
		os.makedirs(self.imagesPath)

		self.imagesList, n = drawImages(origin, number, v=v)

		if v == True:
			print(" [x] Copying {} images from {}".format(n, origin))

		if v == True:
			with progress.Bar(label="    [o] Copying ...", expected_size=len(self.imagesList)) as bar:
				val = 0
				for image in self.imagesList:
					scopy(image, self.imagesPath)
					bar.show(val)
					val += 1
		else:
			for image in imagesList:
				scopy(image, images_dir)

		self.originImagesList = copy.copy(self.imagesList)
		self.imagesList, n = selectImages(self.imagesPath)

	def copyAllImages(self, origin, v=False):
		if self.closed == True:
			print(' [!] This environment is close')
			return

		self.imagesPath = os.path.join(self.name, 'images')
		os.makedirs(self.imagesPath)

		self.imagesList, n = selectImages(origin, v=v)

		if v == True:
			print(" [x] Copying {} images from {}".format(n, origin))

		if v == True:
			with progress.Bar(label="    [o] Copying ...", expected_size=len(self.imagesList)) as bar:
				val = 0
				for image in self.imagesList:
					scopy(image, self.imagesPath)
					val += 1
					bar.show(val)
		else:
			for image in imagesList:
				scopy(image, images_dir)

		self.originImagesList = copy.copy(self.imagesList)
		self.imagesList, n = selectImages(self.imagesPath)

	def copyAllResizedImages(self, origin, y, v=False):
			if self.closed == True:
				print(' [!] This environment is close')
				return

			self.imagesPath = os.path.join(self.name, 'images')
			os.makedirs(self.imagesPath)

			self.pairsPath = os.path.join(self.name, 'pairs')
			os.makedirs(self.pairsPath)

			self.imagesList, n = selectImages(origin, v=v)

			if v == True:
				print(" [x] Copying {} images from {}".format(n, origin))

			if v == True:
				with progress.Bar(label="    [o] Copying ...", expected_size=len(self.imagesList)) as bar:
					val = 0
					for image in self.imagesList:
						img = cv2.imread(image,0)
						res = cv2.resize(img,(614, 1024), interpolation = cv2.INTER_CUBIC)
						equ = cv2.equalizeHist(res)
						cv2.imwrite(os.path.join(self.imagesPath, os.path.basename(image)),equ)
						val += 1
						bar.show(val)
			else:
				for image in imagesList:
					img = cv2.imread(image,0)
					res = cv2.resize(img,(614, 1024), interpolation = cv2.INTER_CUBIC)
					equ = cv2.equalizeHist(res)
					cv2.imwrite(os.path.join(self.imagesPath, os.path.basename(image)),equ)


			self.originImagesList = copy.copy(self.imagesList)
			self.imagesList, n = selectImages(self.imagesPath)


	def removeImages(self, v=False):
		if self.closed == True:
			print(' [!] This environment is close')
			return

		if v == True:
			print(" [x] Removing test environment images")

		if v == True:
			with progress.Bar(label="    [o] Removing ...", expected_size=len(self.imagesList)) as bar:
				val = 0
				for image in self.imagesList:
					os.remove(image)
					val += 1
					bar.show(val)
		else:
			for image in self.imagesList:
				os.remove(image)

	def keepEnv(self, v=False):
		if self.closed == True:
			print(' [!] This environment is close')
			return

		if v == True:
			print(" [x] Keeping a copy of the environment here")

		copytree(self.name, os.path.basename(self.name))

	def writeForBA(numCam, focal, features):

		pass

	def close(self, v=False):
		if self.closed == True:
			print(' [!] This environment is already close')
			return

		if v == True:
			print(" [x] Closing the environment")

		self.closed = True

		rmtree(self.name)





