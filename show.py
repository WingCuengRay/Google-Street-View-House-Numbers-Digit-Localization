# -*- coding:utf8 -*-

import sys
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image

FILE_NAME = 'newCPUPython.txt'
path_prefix = '/home/ray/Code/machine-learning/projects/capstone/data/svhn/full/test/'

class Box:
	def __init__(self, l, t, w, h):
		self.left = l;
		self.top = t;
		self.width = w;
		self.height = h;

# func: 在指定的画布上画一个矩形
# param: 
#    ax --  显示的画布
#    boxes --  一维数组，其元素类型为 NumberBox，该类型包含的矩形的坐标与长宽
def drawRectangle(ax, boxes):
    for i in range(len(boxes)):
        box = boxes[i]
        ax.add_patch(
            patches.Rectangle(
                (box.left, box.top),
                box.width,
                box.height,
                color = 'green',
                fill = False
            )
        )



def readFile(filename):
	file = open(filename)
	all_list = []
	for line in file:
		linelist = line.split(' ')
		cnt = int(linelist[1])

		box_list = []
		for i in range(cnt):
			box_list.append(Box(int(linelist[2+4*i]), int(linelist[3+4*i]), int(linelist[4+4*i]), int(linelist[5+4*i])))
		all_list.append(box_list)

	return all_list




def showSamples(rows, columns, dataset_dir, location):
    plt.close('all')
    fig = plt.figure()

    i = 0
    for row in range(rows):
        for column in range(columns):
            img = Image.open(location[i]['name'])
            ax = fig.add_subplot(rows, columns, i+1)    # 分成 3 行 4 列的子画画布分别显示图片
            drawRectangle(ax, location[i]['box'])
            ax.imshow(img)
            i = i+1       
    plt.show()



def showImage(imgName, filelist):
	seq = int(imgName.split('.')[0])
	img = Image.open(path_prefix+imgName)

	plt.close('all')
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1);
	drawRectangle(ax, filelist[seq-1])
	ax.imshow(img)

	plt.show()



def main():
	if(len(sys.argv) != 2):
		print('Argument error.')
		print('Example: python show.py 1.png')
		exit()

	filelist = readFile(FILE_NAME)
	showImage(sys.argv[1], filelist)

def main_exact_img():
	if(len(sys.argv) != 2):
		print('Argument error.')
		print('Example: python show.py filepath')
		exit()

	filelist = readFile(FILE_NAME)
	img = Image.open(sys.argv[1])
	plt.close('all')
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1);
	drawRectangle(ax, filelist[0])
	ax.imshow(img)

	plt.show()


if __name__ == '__main__':
	#main()
	#main_exact_img()