# -*- coding: utf-8 -*-
import cv2
import os
import math
import numpy as np

address = "/home/ray/Code/machine-learning/projects/capstone/data/svhn/full/test/"
base = "/home/ray/Code/Google-Street-View-House-Numbers-Digit-Localization/cascades/cascade"
outputFile = "./newCPUPython.txt"

#ratios = [1.0, 2.0]  # original ratios
#ratios = [1, 2]  # good for A111_1.jpg
ratios = [0.5, 1]  # good for A203.jpg A206.jpg
imNo = 0
#x_enlarge = 1
#y_enlarge = 1



class Box:
	def __init__(self, l, t, w, h):
		self.left = l;
		self.top = t;
		self.width = w;
		self.height = h;

	# 以 Box 的　x 轴中心点作为比较的　key
	#def __cmp__(self, other):
	#	return (self.left+self.width/2) < (other.left+other.width/2)

	def area(self):
		return self.width * self.height

	def tl(self):
		return (self.left, self.top)

	def br(self):
		return (self.left+self.width, self.top+self.height)


def write(digit_list, confidence, imNo):
	mini = 4
	f = open(outputFile, 'a')

	# sort two list according confidence
	# from small to large
	for i in range(len(digit_list)):
		for j in range(len(digit_list)):
			if confidence[j] < confidence[i]:
				digit_list[i], digit_list[j] = digit_list[j], digit_list[i]
				confidence[i], confidence[j] = confidence[j], confidence[i]


	digit_len = len(digit_list)
	f.write(str(imNo)+' '+str(min(digit_len, mini)))

	# Error?
	# The elements with smaller confidence is in head
	# What we want to get is the element with largest confidence
	for i in range(digit_len):
		format_str = ' ' + str(digit_list[i].left) + \
					 ' ' + str(digit_list[i].top) + \
					 ' ' + str(digit_list[i].width) + \
					 ' ' + str(digit_list[i].height)
		f.write(format_str)
		if i >= mini-1:
			break;
	f.write('\n')
	f.close()

			

# The way of 'sigma' being computed can be replaced by the standard derivative function
def stats(numList):
	length = len(numList)
	sumq = np.sum(numList)
	sumsq = np.sum(np.square(numList))

	mu = sumq/length
	sigma = math.sqrt((sumsq/length)-(mu*mu))

	print(mu, sigma)
	return mu, sigma


def area_filter(all_digits, dist = 0.75):
	wTimesh = []
	for each_digit in all_digits:
		wTimesh.append(each_digit.width * each_digit.height)
	wTimesh = np.array(wTimesh)

	#mu = np.mean(wTimesh)
	#sigma = np.sum(wTimesh)
	mu, sigma = stats(wTimesh)

	filter_digits = []
	for each_digit in all_digits:
		if math.fabs(each_digit.height*each_digit.width - mu) <= (dist*sigma+25):
			filter_digits.append(each_digit)

	return filter_digits


def Overlap(box1, box2):
	# to be continued
	overlap = Box(0, 0, 0, 0)
	if box1.left+box1.width<box2.left or \
		box2.left+box2.width<box1.left:
		return overlap
	if box1.top+box1.height<box2.top or \
		box2.top+box2.height<box1.top:
		return overlap

	left = max(box1.left, box2.left)
	right = min(box1.left+box1.width, box2.left+box2.width)
	top = max(box1.top, box2.top)
	bottom = min(box1.top+box1.height, box2.top+box2.height)

	return Box(left, top, right-left, bottom-top)


def cluster(filter_digits):
	cc = 1.0
	combined_digits = []
	confidence = []
	if len(filter_digits) == 0:
		print('Filtered all digits area!!')
		return combined_digits, confidence

	tmp = filter_digits[0]
	for i in range(1, len(filter_digits)):
		# Unimplemented function
		overlap = Overlap(filter_digits[i], tmp)

		# overlap either half area
		if (overlap.area()>0.5*tmp.area()) or (overlap.area()>0.5*filter_digits[i].area()):
			# recompute the average coordinate
			avg_left = (tmp.left*cc+filter_digits[i].left)//(cc+1)
			avg_top = (tmp.top*cc+filter_digits[i].top)//(cc+1)
			avg_right= ((tmp.left+tmp.width)*cc+(filter_digits[i].left+filter_digits[i].width)) // (cc+1)
			avg_bottom = ((tmp.top+tmp.height)*cc+(filter_digits[i].top+filter_digits[i].height)) // (cc+1)

			# left, top, width, height
			tmp = Box(int(avg_left), int(avg_top), int(avg_right-avg_left), int(avg_bottom-avg_top))
			print(avg_left, avg_top, avg_right-avg_left, avg_bottom-avg_top)
			cc = cc+1
		else:
			combined_digits.append(tmp)
			tmp = filter_digits[i]
			
			# the more rectangles being combined, the 
			# highter the confidence that overlapped area owned
			confidence.append(cc)
			print('Rectangles clustered count: ' + str(cc))
			cc = 1

	combined_digits.append(tmp)
	confidence.append(cc)
	print('Rectangles clustered count: ' + str(cc))

	return combined_digits, confidence




def getAllDigitArea(cascade_list, img, enlarge):
	global imNo
	tmp_copy = img
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_scale = []

	all_digits = []
	for k in range(len(ratios)):
		# y 轴没进行 ratio 缩放
		tmp_img = cv2.resize(img, (0, 0), fx=ratios[k]*enlarge, fy=enlarge)
		img_scale.append(tmp_img)

		for i in range(10):
			#digits = cascade_list[i].detectMultiScale(img_scale[k], 1.1, 3, 
			#	0|cv2.CV_HAAR_SCALE_IMAGE, [20,30], [400, 600])
			digits = cascade_list[i].detectMultiScale(img_scale[k], 1.1, 3, 
				0|cv2.CASCADE_SCALE_IMAGE, (20,30), (400, 600))

			for x,y,width,height in digits:
			#for j in range(len(digits)):
				curr_x = x // ratios[k]*enlarge
				curr_y = y // enlarge  	# y 轴没进行 ratio 缩放
				curr_width = width // (ratios[k]*enlarge);
				curr_height = height // enlarge;
				all_digits.append(Box(int(curr_x), int(curr_y), int(curr_width), int(curr_height)))
				#print(curr_x, curr_y, curr_width, curr_height)
	#all_digits = sorted(all_digits)
	all_digits = sorted(all_digits, key=lambda each_digit : (each_digit.left+each_digit.width/2))
	items = [(b.left, b.top, b.width, b.height) for b in all_digits]
	print(items)
	print('all_digits length: '+str(len(all_digits)))

	if(len(all_digits) > 0):
		filter_digits = area_filter(all_digits)
		print('filter length: '+str(len(filter_digits)))
		cluster_digit, confidence = cluster(filter_digits)
		print('cluster length: '+str(len(cluster_digit)))
		imNo = imNo+1
		write(cluster_digit, confidence, imNo)

	else:
		print('imNo: ', imNo)
		print('Size: ', img.shape[0], img.shape[1])

		# Have probability to cause infinite loop
		getAllDigitArea(cascade_list, tmp_copy, 2*enlarge)


def localize(filepath):
	digit_cascade = []
	for i in range(10):
		filename = base+str(i)+'/cascade.xml'
		digit_cascade.append(cv2.CascadeClassifier(filename))
	print('digit_cascade constructed.')

	img = cv2.imread(filepath)
	getAllDigitArea(digit_cascade, img, 1)




def main():
	digit_cascade = []
	for i in range(10):
		filename = base+str(i)+'/cascade.xml'
		digit_cascade.append(cv2.CascadeClassifier(filename))
	print('digit_cascade constructed.')

	try:
		os.remove(outputFile)
	except:
		pass

	for i in range(1, 101):
		print('Being localize...')
		filename = address + str(i) + '.png'
		img = cv2.imread(filename, 1)
		getAllDigitArea(digit_cascade, img, 1)


#if __name__ == '__main__':
#	main()