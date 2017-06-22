import cv2
import numpy as np
import sys
import os
from fractions import Fraction


def change_background(fg, bg, mask):
	mask_inv = cv2.bitwise_not(mask)
	fg = cv2.bitwise_and(fg, fg, mask = mask)
	bg = cv2.bitwise_and(bg, bg, mask = mask_inv)
	#cv2.imwrite('fg.jpg',fg)
	#cv2.imwrite('bg.jpg',bg)
	res = bg + fg
	return res
	#cv2.imwrite('res.jpg',res)


if __name__ == "__main__":

	if len(sys.argv) < 3:
		print("usage of the script: python changeBackground.py <train.txt> <backgroundpath>")
		exit(0)
	
	with open(sys.argv[1], 'r') as file:
		imageList = [x.split('\n')[0] for x in file.readlines()]

	bgList = []
	for file in os.listdir(sys.argv[2]):
		if file.endswith(".ppm"):
			bgList.append("{}/{}".format(sys.argv[2], file))
	
	trainFile = open(sys.argv[1], 'a')
	for	imagepath in imageList:
		fg = cv2.imread(imagepath, 1)
		print(imagepath)
		savepath = "/".join(imagepath.split('/')[:-1])
		labelpath = "/".join(imagepath.split('/')[:-2])
		labelpath = "{}/labels/".format(labelpath)
		txtname = "/".join(imagepath.split('/')[-1:])
		txtname = "/".join(txtname.split('.')[:-1])
		labelpath = "{}{}".format(labelpath,txtname)
		labelFilePath = "{}.txt".format(labelpath)
		labelFile = open(labelFilePath, 'r')
		bbox = labelFile.readlines()[0]
		pathcut = imagepath.split('.')
		maskpath = '{}_mask.{}'.format(pathcut[0],pathcut[1])
		mask = cv2.imread(maskpath, 0)
		num = 0
		for bgpath in bgList:
			bg = cv2.imread(bgpath, 1)
			#print(bgpath)
			try:
				newimage = change_background(fg, bg, mask)
				#cv2.imshow('new', newimage)
				#cv2.waitKey(5)
				newsavepath = "{}_masked-{}.{}".format(pathcut[0],num,pathcut[1])
				newlabelpath = "{}_masked-{}.txt".format(labelpath,num)
				newLabelFile = open(newlabelpath, 'a')
				newLabelFile.write("{}".format(bbox))
				#print(newlabelpath)
				cv2.imwrite(newsavepath, newimage)
				num += 1
				#print(newsavepath)
				trainFile.write("{}\n".format(newsavepath))
			except:
				print('failed')
				pass
	trainFile.close()
