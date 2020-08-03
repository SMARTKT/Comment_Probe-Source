import os
import csv
import json
import sys

def visitor(filters, dirname, names):
	#print(dirname)
	mynames = filter(lambda n : os.path.splitext(n)[1].lower() in filters, names)
	for name in mynames:
		fpath = os.path.join(dirname, name)
		if not os.path.isdir(fpath):
			absolute_path = fpath[fpath.find(sys.argv[5])+len(sys.argv[5]):]
			if absolute_path[0] == "/":
				absolute_path = absolute_path[1:]
			xmlpath = sys.argv[4] + absolute_path[absolute_path.find("/") + 1: ]
			xmlpath = xmlpath.replace(".cpp", "_clang.xml").replace(".cc","_clang.xml").replace(".c", "_clang.xml")
			print(fpath.find(sys.argv[5]),fpath, sys.argv[2], sys.argv[3], xmlpath, absolute_path)
			#break
			os.system("python2 XML_GetExcelForAFile.py " + fpath + " " + sys.argv[2] + " " + sys.argv[3] + " " +  xmlpath + " " + absolute_path)

if __name__ == "__main__":
	if len(sys.argv) != 6:
		print("Give 5 arguments: source code folder location, vocab dictionary location, problem domain location, xml folder location, project folder exact name - in this order")
		exit(-1)
	filters = [".c", ".cpp", ".cc"]
	os.path.walk(sys.argv[1], visitor, filters)