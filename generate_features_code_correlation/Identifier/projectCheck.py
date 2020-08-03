import MySQLdb
import os
from glob import glob
import csv
from contextlib import closing
import sys

PATH = '/home/srijoni/Desktop/LLVM_Install/llvm/build/input_files/go/'
cc_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.cc'))]
cpp_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.cpp'))]
c_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.c'))]
#h_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.h'))]

#files = c_files + cc_files + cpp_files
file = '/home/srijoni/Desktop/LLVM_Install/llvm/build/virtual.cpp'
files = [file]

#print(files)
print('-----------------')

f = open('working files','a')
count = 0
for file in files:
	#print(file)
	if count == 15:
		print("success")
		break
	exe = "../bin/syntax-tree " + str(file)
	os.system(exe)

	conn = MySQLdb.connect(host = 'localhost', user = 'root', passwd = 'srijoni123')  # create the connection
	with closing(conn.cursor()) as cursor:
		cursor.execute("USE test;")
		cursor.execute("USE test;")
		cursor.execute("SHOW TABLES") 
		tables = cursor.fetchall()       # return data from last query
		for table in tables:
			print(table)
			cursor.execute("select * from " + str(table[0]) + "\n")
			rows = cursor.fetchall()
			print(rows)
			if len(rows) > 0:
				f.write(file)
				count +=1
				break

f.close()
