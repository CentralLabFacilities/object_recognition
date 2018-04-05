import cv2
import numpy as np
import sys
import os
from random import randint
from fractions import Fraction
from image_recognition_util.objectset_utils import ObjectsetUtils
from image_recognition_util.object import BoundingBox
from object_detection.utils import label_map_util

def getRandomPositionOnSurface(bbox, bgAnnotationList):
	# randomly choose one surface annotation
	l = len(bgAnnotationList)
	r = randint(0,l-1)
	surfaceBox = bgAnnotationList[r].bbox

	# get possible range to place roi
	roiW = (bbox.xmax - bbox.xmin)
	roiH = (bbox.ymax - bbox.ymin)

	limitLeft = surfaceBox.xmin + roiW
	limitRight = surfaceBox.xmax
	limitDown = surfaceBox.ymax
	limitUp = max(roiH,surfaceBox.ymin)

	if (limitRight < limitLeft or limitUp > limitDown):
		print("Surface too small to place roi.")
		return None

	bboxRand = BoundingBox(0,0,0,0)

	# choose random point in surface range (bottom of object)
	bboxRand.ymax = randint(limitUp, limitDown)
	bboxRand.xmax = randint(limitLeft, limitRight)

	# set other coordinates according to roi size
	bboxRand.xmin = bboxRand.xmax - roiW
	bboxRand.ymin = bboxRand.ymax - roiH

	return bboxRand

def placeRoiOnBackground(fgCut, maskCut, bg, bbox):

	#cut out object and background based on mask inside the roi
	maskInv = cv2.bitwise_not(maskCut)
	fg = cv2.bitwise_and(fgCut, fgCut, mask=maskCut)
	try:
		bgCut = bg[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
		bgCut = cv2.bitwise_and(bgCut, bgCut, mask=maskInv)
		roi = bgCut + fg
		#insert roi in large image
		h, w, c = bg.shape
		new = np.zeros((h, w, c), np.uint8)
		new[0:h, 0:w] = bg
		new[bbox.ymin:bbox.ymin + roi.shape[0], bbox.xmin:bbox.xmin + roi.shape[1]] = roi
		return new
	except:
		print("exception with mask-shape: {} and bg-shape: {}".format(maskCut.shape, bgCut.shape))

def fillImgList(list, dir):
	for dirname, dirnames, filenames in os.walk(dir):
		for filename in filenames:
			file = dirname + '/' + filename
			if file.endswith(".jpg"):
				list.append("{}".format(file))

def findLabelMap(dir):
	for dirname, dirnames, filenames in os.walk(dir):
		for filename in filenames:
			file = dirname + '/' + filename
			if file.endswith(".pbtxt"):
				labelMap = label_map_util.load_labelmap(file)
				print "loading {}".format(file)
				return labelMap
	print "error: No labelMap found. Exiting."
	exit(1)


def show(img,time):
	cv2.imshow('img', img)
	cv2.waitKey(time)

def createSaveDirectoryByObjectCategories(labelMap, saveDir):
	# create save directories for all object categories
	categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=99,
																use_display_name=True)
	categoryIndex = label_map_util.create_category_index(categories)
	for i in range(0, len(categoryIndex)):
		classText = categoryIndex[i + 1]['name']
		imgDir = saveDir + '/' + classText + '/images'
		lblDir = saveDir + '/' + classText + '/labels'
		if not os.path.exists(imgDir):
			os.makedirs(imgDir)
		if not os.path.exists(lblDir):
			os.makedirs(lblDir)

if __name__ == "__main__":

	if len(sys.argv) < 5:
		print("usage: python changeBackground.py <annotation_dir> <background_dir> <save_dir> <use_bg_n_times>")
		exit(1)

	if not os.path.isdir(sys.argv[1]):
		print '\033[91m' + sys.argv[1] + ' is not a directory!' + '\033[0m'
		exit(1)
	if not os.path.isdir(sys.argv[2]):
		print '\033[91m' + sys.argv[2] + ' is not a directory!' + '\033[0m'
		exit(1)
	n = int(sys.argv[4])
	if (n <= 0):
		print "invalid number: {}".format(sys.argv[4])
		exit(1)

	annotationDir = sys.argv[1]
	bgDir = sys.argv[2]
	saveDir =sys.argv[3]
	if not os.path.exists(saveDir+'/darkset'):
		print("create save dir: {}".format(saveDir+'/darkset'))
		os.makedirs(saveDir+'/darkset')
	
	annotationList = []
	fillImgList(annotationList,annotationDir)
	labelMap = findLabelMap(annotationDir)
	bgLabelMap = findLabelMap(annotationDir)

	createSaveDirectoryByObjectCategories(labelMap, saveDir+'/darkset')

	bgList = []
	fillImgList(bgList,bgDir)

	util = ObjectsetUtils()

	# start background changing
	for	imgPath in annotationList:
		labelPath = imgPath.replace("/images/", "/labels/").replace(".jpg", ".txt")
		maskPath = imgPath.replace(".jpg", "_mask.jpg")
		# only proceed if label and mask file exist
		if not (os.path.isfile(labelPath) and os.path.isfile(maskPath)):
			print("error: Label file or mask file does not exist! Skipping image.")
			continue

		# TODO: get num_classes(=99) from labelMap
		labelList = util.readAnnotated(labelPath, labelMap, 99)

		#get masked object (roi only)
		fg = cv2.imread(imgPath,1)
		mask = cv2.imread(maskPath, 0)

		# use smaller mask and set all values to 0 or 255
		kernel = np.ones((5, 5), np.uint8)
		#TODO: best number of iterations?
		maskErosion = cv2.erode(mask, kernel, iterations=3)
		for (x,y), value in np.ndenumerate(maskErosion):
			if maskErosion[x][y] > 200:
				maskErosion[x][y] = 255
			else:
				maskErosion[x][y] = 0

		h, w, c = fg.shape

		for label in labelList:
			# cut roi from image and mask
			bbox = util.getAbsoluteRoiCoordinates(label.bbox,w,h)
			fgCut = fg[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
			maskCut = maskErosion[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
			for bgpath in bgList:
				print bgpath
				bgLabelPath = bgpath.replace("/images/", "/labels/").replace(".jpg", ".txt")
				# only proceed if label and mask file exist
				if not (os.path.isfile(bgLabelPath)):
					print("error: Label file does not exist! Skipping image.")
					continue
				bg = cv2.imread(bgpath, 1)
				#TODO: replace 99 -> numClasses
				bgAnnotationList = util.readAnnotated(bgLabelPath, bgLabelMap, 99)
				for i in range(0,len(bgAnnotationList)):
					bgAnnotationList[i].bbox = util.getAbsoluteRoiCoordinates(bgAnnotationList[i].bbox, w, h)
				# place roi on n random positions in background image
				for i in range(0,n):
					bboxRand = getRandomPositionOnSurface(bbox, bgAnnotationList)
					if not bboxRand:
						continue
					newImg = placeRoiOnBackground(fgCut, maskCut, bg, bboxRand)

					saveImg = imgPath.replace(annotationDir,saveDir).replace(".jpg", "_bg{}.jpg".format(str(i)))
					saveLabels = saveImg.replace("/images/", "/labels/").replace(".jpg", ".txt")
					# save image
					cv2.imwrite(saveImg, newImg)
					# save label
					idList = []
					idList.append(label.label)
					boxList = []
					boxList.append(bboxRand)
					util.writeAnnotationFile(saveLabels, idList, boxList, bg)

	print "done"
