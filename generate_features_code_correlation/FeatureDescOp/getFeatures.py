import csv
import sys
import re

def readCSV(fname):
	f = open(fname, 'r')
	r = csv.reader(f)
	out = []
	for each in r:
		out.append(each)
	f.close()
	return out

def writeCSV(fname, output):
	outfile = open(fname, 'wb')
	writer = csv.writer(outfile , delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
	for eachrow in output: 
		writer.writerow(eachrow)
	outfile.close()

def getCount(s, d):
	s = s.replace(" ", "")
	if s == "":
		return 0
	return len(s.split(d))


def numDescriptional(s):
	s = s.split("|")
	num = 0
	for each in s:
		now = each.replace(" ", "")
		if now == "":
			continue
		if now == "d":
			num += 1
	return num

def numOperational(s):
	s = s.split("|")
	num = 0
	for each in s:
		now = each.replace(" ", "")
		if now == "":
			continue
		now = now.split(":")
		if now[0] == "o":
			num += 1
	return num

def getIdTypeInfo(s):
	s = s.split(" |||")
	ans = [0, 0, 0]
	for each in s:
		now = each.replace(" ", "")
		if now == "":
			continue
		now = now.split(":")
		if now[0].endswith("Function"):
			ans[0] += 1
		elif now[0].endswith("Stmt"):
			ans[1] += 1
		else:
			ans[2] += 1
	return ans

def getCommentLines(s):
	s = s.split(":")
	print s
	return int(s[1]) - int(s[0]) + 1

def getLibraryCopyrightInfo(text):
	ans = [0, 0]
	text = text.lower()
	if text.find("copyright") != -1:
		ans[0] = 1
	if text.find(".h") != -1 or text.find("#include") != -1:
		ans[1] = 1
	return ans

def getFeatures(fname):
	data = readCSV(fname)
	features = [["#comment tokens", "#prog. conc", "#prob. conc", "descriptive", "operative", "conditional", "#scope_symbols", "#scope_functions", "scope_compound_statement1s", "scope_param", "scope_line_numbers", "prog_dom_match_identifiers", "prob_dom_match_identifiers", "copyright_license_comments", "libraries"]]
	for i in range(1, len(data)):
		line = data[i]
		feature = []
		feature.append(getCount(line[3], "|||"))
		feature.append(getCount(line[5], "]["))
		feature.append(getCount(line[6], "|||"))
		feature.append(numDescriptional(line[16]))
		feature.append(numOperational(line[16]))
		feature.append(len(re.findall("True", line[17])))
		feature.append(getCount(line[7], "|||"))
		type_info = getIdTypeInfo(line[8])
		feature.extend(type_info)
		feature.append(getCommentLines(line[4]))
		feature.append(getCount(line[13], "|||"))
		feature.append(getCount(line[14], "|||"))
		libcopyinfo = getLibraryCopyrightInfo(line[2])
		feature.extend(libcopyinfo)
		
		features.append(feature)
	
	ofname = fname.replace("excel", "feature")
	writeCSV(ofname, features)
	
	
if len(sys.argv) != 2:
	print "Give one argument which is the location of knowledge base csv"
	exit()

fname = sys.argv[1]
getFeatures(fname)
