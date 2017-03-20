from detect import localize
import sys
import os

def main():
	if len(sys.argv) != 2:
		print('Format: python localizate.py filePath')
		exit()

	os.remove('./newCPUPython.txt')
	localize(sys.argv[1])

if __name__ == '__main__':
	main()