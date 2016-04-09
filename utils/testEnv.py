import os
import time

from shutil import copy, move, copytree, rmtree

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
					copy(image, self.imagesPath)
					bar.show(val)
					val += 1
		else:
			for image in imagesList:
				copy(image, images_dir)

		self.originImagesList = self.imagesList.copy()
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
					copy(image, self.imagesPath)
					bar.show(val)
					val += 1
		else:
			for image in imagesList:
				copy(image, images_dir)

		self.originImagesList = self.imagesList.copy()
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
					bar.show(val)
					val += 1
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

	def close(self, v=False):
		if self.closed == True:
			print(' [!] This environment is already close')
			return

		if v == True:
			print(" [x] Closing the environment")

		self.closed = True

		rmtree(self.name)





