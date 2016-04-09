import random
import os

"""
fileAndImage
----------
Verify if the file is really a file and an image

filepath (string): path of the file
good_ext (optional - tuple/list of string) : good extension to take as an image
v (optional-boolean): verbose mode

return (boolean): False if not, True if is 
"""
def fileAndImage(filepath, good_ext=None, v=False):

	if not isinstance(filepath, str):
		print(' [!] filepath must be a string')
		raise TypeError

	if good_ext is None:
		good_ext = (".jpg", ".JPG")

	if not isinstance(good_ext, tuple) and not isinstance(good_ext, list):
		print(' [!] good_ext must be a tuple or a list')
		raise TypeError

	if not os.path.isfile(filepath):
		if v == True:
			print(' [x] {} is not a file'.format(filepath))
		return False

	_, ext = os.path.splitext(filepath)

	if not ext in good_ext:
		if v == True:
			print(' [x] {} is not an image'.format(filepath))
		return False

	if v == True:
		print(' [x] {} is a file and an image'.format(filepath))

	return True


"""
drawImages
----------
Randomly draw images from a directory

dirpath (string): directory path
number (integer): number to draw
v (optional - boolean): verbose mode

return (tuple): (list of string) list of images kept, (integer) number of images drawn
"""
def drawImages(dirpath, number, v=False):

	if not isinstance(dirpath, str):
		print(' [!] dirpath must be a string')
		raise TypeError

	if not isinstance(number, int) or number < 0:
		print(' [!] number must be an integer > 0')
		raise TypeError

	if v == True:
		print(" [x] Randomly drawing {0} images of {1}".format(number, dirpath))

	images = [os.path.join(dirpath, im) for im in os.listdir(dirpath)\
	if fileAndImage(os.path.join(dirpath, im))]

	try:
		idx = random.sample(range(0,len(images)), int(number))
	except ValueError:
		if v == True:
			print(" [!] Not enough images, will take all images.")
		number = len(images)
		idx = random.sample(range(0,len(images)), int(number))

	images_kept = [im for i, im in enumerate(images) if i in idx]

	return (images_kept, number)

"""
selectImages
----------
select images from a directory

dirpath (string): directory path
v (optional - boolean): verbose mode

return (tuple): (list of string) list of images, (integer) number of images
"""
def selectImages(dirpath, v=False):

	if not isinstance(dirpath, str):
		print(' [!] dirpath must be a string')
		raise TypeError

	if v == True:
		print(" [x] Selecting images of {}".format(dirpath))

	images = [os.path.join(dirpath, im) for im in os.listdir(dirpath)\
	if fileAndImage(os.path.join(dirpath, im))]

	if v == True:
		print("    [o] {} images found".format(len(images)))

	return (images, len(images))


"""
generateImagesList
----------
Generate an images list from a list of images

dirpath (string): directory path
images (list of string): list of images path
v (optional - boolean): verbose mode

return nothing
"""
def generateImagesList(dirpath, images, v=False):

	if not isinstance(dirpath, str):
		print(' [!] dirpath must be a string')
		raise TypeError

	if not isinstance(images, tuple) and not isinstance(images, list):
		print(' [!] images must be a tuple or a list')
		raise TypeError

	filename = os.path.join(dirpath, "images.txt")

	if v == True:
		print(" [x] Generating an images list in {}".format(dirpath))

	with open(filename, "w") as f:
		f.write("#"*21+"\n")
		f.write("#"+" "*4+"Images used"+" "*4+"#"+"\n")
		f.write("#"*21+"\n")
		f.write("\n".join(images))