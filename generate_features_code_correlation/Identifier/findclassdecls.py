import MySQLdb
import os
import csv
import sys
from contextlib import closing

os.system('rm -r stats')

os.system('mkdir stats')


def getProgramDomainWords(fname):
	f = open(fname, 'r')
	prog_dom_reader = csv.reader(f)
	dom_list = {}
	for row in prog_dom_reader:
		words = row[0].split(" ")
		for each in words:
			dom_list[each.lower()] = row[1]
	f.close()
	return dom_list

def getProblemDomainWords(fname):
	f = open(fname, 'r')
	text = f.read()
	text = text.split("\n")
	dom_list = {}
	for row in text:
		dom_list[row.lower()] = "ProblemDomain"
	f.close()
	return dom_list

def wordSegmentation(word):
	words = word.split("_")
	output = []
	for each in words:
		# Split if there is a change in lower to uppercase.
		# Eg: printArray is split as print, Array		
		i = 0
		j = 0
		while j < len(each):
			if j == len(each) - 1:
				output.append(each[i:j + 1])
				i = j + 1
			else:
				if each[j].islower() and each[j+1].isupper():
					output.append(each[i:j + 1])
					i = j + 1
			j += 1
	return output

def findProgramDomainMatches(tokens):
	vocab_dict = getProgramDomainWords(sys.argv[2])	
	output = ""
	for each in tokens:
		now = each.lower()
		if now in vocab_dict:
			output += now + " : " + vocab_dict[now] + " | "
	return output[:-3]
 

def findProblemDomainMatches(tokens):
	prob_dict = getProblemDomainWords(sys.argv[2])	
	output = ""
	for each in tokens:
		now = each.lower()
		if now in prob_dict:
			output += now + " | "
	return output[:-3]

def joinBySpace(l):
	if len(l) == 0:
		return ""
	ans = ""
	for each in l:
		ans += each + " "
	return ans[:-1]

# NOTE: This code is not written for C files which have interdependencies in them. findclassdecls fails on the C files with dependencies.
# Given a C file as input, we will generate a CSV file with the fields:
# symbol : The identifier symbol written in the code
# line_begin : start line of the identifier's scope
# line_end : end line of the identifier's scope
# data_type : data_type if it is a variable.
# Program domain matches : Program domain matches if found any
# Problem domain matches : Problem domain matches if found any

def main():
	if len(sys.argv) != 4:
		print "Give 3 arguments (filename, program domain location, problem domain location) in order."
		exit(-1)
	fname = sys.argv[1]
	fname_small = fname.replace('.', '_')
	fname_small = fname_small.split('/')
	fname_small = fname_small[-1]
	
	fname = fname_small
	os.system('mkdir stats/' + fname)
	conn = MySQLdb.connect(host = 'localhost', user = 'root', passwd = 'srijoni321')  # create the connection
	with closing(conn.cursor()) as cursor:
		exe = "../../../../../home/srijoni/spandan/llvm_build_copy/build/bin/find-class-decls " + sys.argv[1]
		os.system(exe)
		cursor.execute("use test;")
		cursor.execute("show tables;") 
		tables = cursor.fetchall()
		print "tables", tables
		output = []
		for table in tables:
			query = "select symbol, type, line_begin, line_end, data_type from " + str(table[0])
			cursor.execute(query)
			results = cursor.fetchall()
			for each in results:
				symbol_now = each[0]
				if symbol_now != None:
					tokens = wordSegmentation(symbol_now)
					prog_matches = findProgramDomainMatches(tokens)
					prob_matches = findProblemDomainMatches(tokens)
				else:
					tokens = []
					prog_matches = ""
					prob_matches = ""
				output.append([each[0], each[1], each[2], each[3], each[4], joinBySpace(tokens), prog_matches, prob_matches])
		f = open("identifiers.csv", 'w')
		writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(["Symbol", "Type", "Start line", "End line", "Data type", "Identifier tokens", "Program Domain matches", "Problem Domain matches"])
		for each in output:
			writer.writerow(each)
		f.close()
main()

