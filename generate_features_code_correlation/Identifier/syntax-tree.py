import MySQLdb
import os
from glob import glob
import csv
from contextlib import closing
import sys

import re
import pickle
import subprocess


os.system('rm -r stats')

os.system('mkdir stats')

classes = []
global_functions = [] # main is a global function
all_variables = []
global_var = []

methods = []
param = {}
decl_var = {}
fields = {}

def write_csv(data, file_name):
	c = csv.writer(open("stats/" + file_name, "a+"))
	for row in data:
			c.writerow(row)

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def extract_param(table_name, file_name):
	conn = MySQLdb.connect(host = 'localhost', user = 'root', passwd = 'srijoni321')  # create the connection
	print(table_name)
	with closing(conn.cursor()) as cursor:
		cursor.execute("USE test;")
		query = "select symbol, parent, data_type, line_begin, line_end, accSpec, isOverload, isVirtual, isStatic, isInline, \
		        isExtern, isConstant  from " + table_name + " where type = 'ParmVar';"
		cursor.execute(query)
		result = cursor.fetchall()
		write_csv(result, file_name + "/params.csv")
		result = [res[0] for res in result]
	conn.close()
	return result


def extract_decl_var(table_name, file_name):
	conn = MySQLdb.connect(host = 'localhost', user = 'root', passwd = 'srijoni321')  # create the connection
	with closing(conn.cursor()) as cursor:
		cursor.execute("USE test;")
		query = "select child from " + table_name + " where type = 'CompoundStmt';"
		cursor.execute(query)
		result = cursor.fetchall()
		query = "select symbol, parent, data_type, line_begin, line_end, accSpec, isOverload, isVirtual, isStatic, isInline, \
		        isExtern, isConstant  from " + result[0][0] + " where type = 'Var';"
		cursor.execute(query)
		result = cursor.fetchall()
		write_csv(result, file_name + "/decls.csv")
		result = [res[0] for res in result]
	conn.close()
	return result

def get_class_fields(table_name, file_name):
	conn = MySQLdb.connect(host = 'localhost', user = 'root', passwd = 'srijoni321')  # create the connection
	with closing(conn.cursor()) as cursor:
		cursor.execute("USE test;")
		query = "select symbol, parent, data_type, line_begin, line_end, accSpec, isOverload, isVirtual, isStatic, isInline, \
		        isExtern, isConstant  from " + table_name + " where type = 'Field of " + table_name[:-1-len(file_name_substring)] + "';"
		cursor.execute(query)
		result = cursor.fetchall()
		write_csv(result, file_name + "/fields.csv")
		fields[table_name[:-1-len(file_name_substring)]] = [res[0] for res in result]
	conn.close()

def get_methods(table_name, file_name):
	conn = MySQLdb.connect(host = 'localhost', user = 'root', passwd = 'srijoni321')  # create the connection
	with closing(conn.cursor()) as cursor:
		cursor.execute("USE test;")
		query = "select symbol, parent, data_type, line_begin, line_end, accSpec, isOverload, isVirtual, isStatic, isInline, \
		        isExtern, isConstant  from " + table_name + " where type = 'CXXMethod'"
		cursor.execute(query)
		result = cursor.fetchall()
		write_csv(result, file_name + "/methods.csv")
	conn.close()

# if len(sys.argv) == 1:
# 	cmd = 'find-class-decls '
# 	num = 1
# 	print('Project and cmd unspecified, default values are 1, find-class-decls')
# elif len(sys.argv) == 2:
# 	num = sys.argv[1]
# 	cmd = 'find-class-decls '
# 	print('Cmd unspecified, default cmd is find-class-decls')
# elif len(sys.argv) == 3:
# 	num = sys.argv[1]
# 	if sys.argv[2] == 'fcl':
# 		cmd = 'find-class-decls '
# 	elif sys.argv[2] == 'st':
# 		cmd = 'syntax-tree '
# 	else:
# 		print('Invalid arguments')
# 		cmd = 'find-class-decls '
# 		num = 1
# 		print('Using default values are 1, find-class-decls')
		

# PATH = '/home/srijoni/Desktop/LLVM_Install/llvm/build/input_files/' + str(num) + '/'
# cc_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.cc'))]
# cpp_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.cpp'))]
# c_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.c'))]
# h_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.h'))]

# files = c_files + cc_files + cpp_files

files = sys.argv[1:]

# problem_domain = []
# with open('probdom' + str(num) + '.txt') as f1:
#     problem_domain = f1.read().splitlines()

# program_domain = {}
# reader = csv.reader(open('program_domain.csv','r'))
# program_domain = dict(reader)


cnt = 1;
cnt_progdom = 0;
cnt_probdom = 0;

all_tokens = {}



for file in files:
	print(file)
	file_name_substring = str(file).replace('.', '_')

	file_name_substring = file_name_substring.split('/')
	file_name_substring = file_name_substring[-1]
	file_name = file_name_substring
	os.system('mkdir stats/' + file_name)
	conn = MySQLdb.connect(host = 'localhost', user = 'root', passwd = 'srijoni321')  # create the connection
	with closing(conn.cursor()) as cursor:
		exe = "../../../../../home/srijoni/spandan/llvm_build_copy/build/bin/syntax-tree " + files[0] 
		os.system(exe)
		#subprocess.call(["../bin/syntax-tree", str(file)])
		cursor.execute("USE test;")
		cursor.execute("SHOW TABLES") 
		tables = cursor.fetchall()       # return data from last query
		for table in tables:
			query = "show columns from " + table[0] + ";"
			cursor.execute(query)
			result=cursor.fetchall()
			result = [tup[0] for tup in result]
			if 'attrib' in result:
				query = "select attrib from " + table[0] + ";"
				cursor.execute(query)
				result=cursor.fetchall()
				token = table[0][:-1-len(file_name_substring)]
				if len(result) != 0:
					if result[0][0] == 'class': #classes
						classes.append(token)
						get_class_fields(table[0], file_name)
						get_methods(table[0], file_name)
					elif result[0][0] == 'function': 
						arr = token.split('_')
						if arr[-1] in classes: #relies on the fact that classes recognised before their methods
							class_name = arr[-1]
							func_name = '_'.join(arr[:-1])
							#print (func_name + 'in class ' + arr[-1])
							method = (func_name, class_name)
							methods.append(method)
							param[token] = extract_param(table[0], file_name)
							decl_var[token] = extract_decl_var(table[0], file_name)
						elif token[:5] != 'table': #global functions and main function
							global_functions.append(token)
							param[token] = extract_param(table[0], file_name)
							decl_var[token] = extract_decl_var(table[0], file_name)
				# 	else:
				# 		has_attrib.append(token)
				# else:
				# 	no_attrib.append(token)

		#global_var
		cursor.execute("select symbol, parent, data_type, line_begin, line_end, accSpec, isOverload, isVirtual, isStatic, isInline, \
		                isExtern, isConstant  from global_" + file_name_substring + " where type = 'Var';")
		result = cursor.fetchall()
		write_csv(result, file_name + "/global_var.csv")
		global_var = [res[0] for res in result]

		#global_func
		cursor.execute("select symbol, parent, data_type, line_begin, line_end, accSpec, isOverload, isVirtual, isStatic, isInline, \
		                isExtern, isConstant  from global_" + file_name_substring + " where type = 'Function';")
		result = cursor.fetchall()
		write_csv(result, file_name + "/global_func.csv")

		#classes
		cursor.execute("select symbol, parent, data_type, line_begin, line_end, accSpec, isOverload, isVirtual, isStatic, isInline, \
		                isExtern, isConstant  from global_" + file_name_substring + " where type = 'CXXRecord';")
		result = cursor.fetchall()
		write_csv(result, file_name + "/classes.csv")

		cursor.execute("select * from inheritance_table;")
		result = cursor.fetchall();
		write_csv(result, file_name  + "/inheritance.csv")


		#TAKE CARE OF METHODS AND THEN TOKEN TO TOKEN RELATIONS
		#parent isnt reliable, it stores weird values for for example

		#print('global variables are' + str(global_var))
		#all_variables = [item for var in param for item in param[var]] + [item for var in decl_var for item in decl_var[var]] + global_var + fields
		#print(param)
		#print(decl_var)
		#print(all_variables)
		#print(global_functions)
		#print(fields)

	with open("stats/" + file_name + "/methods.txt", "wb") as fp1:   #Pickling
		pickle.dump(methods, fp1)

	with open('stats/' + file_name + '/param.txt', 'wb') as fp2:
		pickle.dump(param, fp2)

	with open('stats/' + file_name + '/decl_var.txt', 'wb') as fp3:
		pickle.dump(decl_var, fp3)

	with open('stats/' + file_name + '/fields.txt', 'wb') as fp4:
		pickle.dump(fields, fp4)

	conn.close()


#var - params, decl inside function tables(methods, main, global), global var, CLASS FIELDS
		
