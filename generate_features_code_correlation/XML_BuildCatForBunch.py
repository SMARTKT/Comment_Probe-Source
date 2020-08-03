import os
import csv
import json
import sys

def visitor(files):
	for fpath in files:
		excel = fpath
		feature = excel.replace("excel", "feature")
		os.system("python2 XML_getFeatures.py " + excel)
		os.system("python2 XML_getFinalFeatures.py " + excel + " " + feature)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Give 1 argument: location of file with all the excel files to be run")
		exit(-1)
	f = open(sys.argv[1], 'r')
	txt = f.read()
	txt = txt.split("\n")
	files = []	
	for each in txt:
		if each != "":
			files.append(each)
	visitor(files)
