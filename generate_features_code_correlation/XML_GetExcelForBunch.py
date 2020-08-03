import os
import csv
import json
import sys

def visitor(files):
	for fpath in files:
		absolute_path = fpath[fpath.find(sys.argv[4]):]
		xmlpath = sys.argv[3] + absolute_path[absolute_path.find("/") + 1: ]
		xmlpath = xmlpath.replace(".c", "_clang.xml")
		os.system("python2 XML_GetExcelForAFile.py " + fpath + " " + sys.argv[1] + " " + sys.argv[2] + " " +  xmlpath + " " + absolute_path)

if __name__ == "__main__":
	if len(sys.argv) != 6:
		print("Give 5 arguments: vocab dictionary location, problem domain location, xml folder location, project folder exact name - in this order and file with all the files to be run")
		exit(-1)
	f = open(sys.argv[5], 'r')
	txt = f.read()
	txt = txt.split("\n")
	files = []	
	for each in txt:
		if each != "":
			files.append(each)
	visitor(files)
