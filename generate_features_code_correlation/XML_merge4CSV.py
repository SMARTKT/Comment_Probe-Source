# 389 399 150

import csv
import sys
def readCSV(fname, x):
	f = open(fname, 'r')
	r = csv.reader(f, delimiter = ',', quotechar = '"' , quoting = csv.QUOTE_MINIMAL)
	out = []
	for each in r:
		out.append(each)
	f.close()
	return out[:x]

if len(sys.argv) != 9:
	print "Give 4 csv files as arguments and 4 line numbers as arguments"
	exit(-1)

data1 = readCSV(sys.argv[1], int(sys.argv[5]))
data2 =	readCSV(sys.argv[2], int(sys.argv[6]))
data3 =	readCSV(sys.argv[3], int(sys.argv[7]))
data4 = readCSV(sys.argv[4], int(sys.argv[8]))

data1.extend(data2)
data1.extend(data3)
data1.extend(data4)

f = open("Models/training.csv", 'w')
writer = csv.writer(f, delimiter=',', quotechar = '"', quoting=csv.QUOTE_NONNUMERIC)
for each in data1:
	writer.writerow(each)
f.close()
