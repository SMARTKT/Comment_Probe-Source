import os
import sys
# Code to run a file to generate the graph

if len(sys.argv) < 2:
	print ("Please give the filename as the argument")
	exit(-1)

fname = sys.argv[1]

import os.path
from os import path

if path.isfile(fname) == False:
	print ("Please give a file which is present")
	exit(-1)

if fname.endswith(".c") == False:
	print ("Please give a C file as an input")
	exit(-1)

# Step1: Run Syntax Tree
try:
	os.system("python2 syntax-tree.py " + fname)
except Exception as e:
	print ("Error in running syntax-tree " + e)
	exit(-1)

# Step2: Run graphgen
try:
	os.system("python2 graphgen.py " + fname)
except Exception as e:
	print("Error in running graphgen " + e)
	exit(-1)

exit(0)
