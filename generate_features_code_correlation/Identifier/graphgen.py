import pickle
import re
import sys

all_nodes = {}

concept_nodes = [] #(nodeid, name, style, concept_type)
token_nodes = [] #(nodeid, name, token_type, startline, endline) 
relations = [] #(nodeid1, nodeid2, relation_type)

program_domain = {} #mergesort - algorithm and so on
problem_domain =  []


with open('problem_domain.txt') as f:
    problem_domain = f.read().splitlines()

with open('program_domain.csv') as f:
	for line in f:
		arr = line.split(',')
		program_domain[arr[0]] = arr[1][:-1].replace('_',' ')


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


# 1 all upper
# 2 all lower
# 3 starting capital
# 4 trailing underscore
# 5 leading underscore
# 6 underscore split
# 7 camelcase
def get_styles(string):
	styles = []
	if string.isupper():
		styles.append(1)
	elif string.islower():
		styles.append(2)
	elif string[0].isupper() and string[1:].islower():
		styles.append(3)

	if string.endswith('_'):
		styles.append(4)

	if string.startswith('_'):
		styles.append(5)

	string = string.strip('_')
	arr = string.split('_')
	#print(arr)
	if len(arr) > 1:
		styles.append(6)
	elif len(arr) == 1:
		arr = camel_case_split(string)
		if len(arr) > 1:
			styles.append(7)

	return styles

def get_all_tokens(string):
	string = string.strip('_')
	arr = string.split('_')
	if len(arr) > 1:
		return arr
	elif len(arr) == 1:
		arr = camel_case_split(string)
		if len(arr) > 1:
			return arr
	return [string]

def prog_tokens(string):
	arr = get_all_tokens(string)
	program_tokens = []
	for token in arr:
		if token in program_domain:
			program_tokens.append(token)
	return program_tokens

def prob_tokens(string):
	if('cpp' in string):
		string = string[:-11]
	#print(string)
	arr = get_all_tokens(string)
	#print('all tokens ' + str(arr))
	problem_tokens = []
	for token in arr:
		if token.lower() in problem_domain:
			problem_tokens.append(token)
		#else:
			#print(token + 'not in probdom')
	return problem_tokens


# print(program_domain)
# print(problem_domain)

# def isProgramDomain(string):
# 	#check if any component token matches a program domain word

# def isProblemDomain(string):
# 	#check if any component token matches a problem domain word

#nodeid - int
#name - string 
#concept_type - string, one of {program domain term, problem domain term, modifier*}
# * modifiers include one of -  global, extern, inline, overload, virtual, public/private, static, const
#token_type - class, variable, function

#location - string file
#start line int line_number
#end line
#relation_type - 1. is a , 2. param, 3. fieldOf, 4. member of, 5. derives 

#relations becomes edgelist during visualisation


# token to concept (8 modifiers, progdom, probdom)
# token to token (param, fieldOf, member of, derives)
	#fields to class
	#param, decl_var to method
	#method to class
	#class to class


#later todos:
#friend keyword
#perhaps add datatype to it
#file excluded from the structure for now, will also be a token node field

#tokens


fname = str(sys.argv[1]).replace('.', '_')
fname = fname.split('/')
fname = fname[-1]
af = fname.replace(".", "_") + "/"
fi = [0, 0, 0, 0, 0, 0]
err = [0, 0, 0, 0, 0, 0]

try:
	fi[0] = open("stats/" + af + "global_var.csv")
except:
	print("1. Global var csv not present")
	err[0] = 1

try:
	fi[1] = open("stats/" + af + "global_func.csv")
except:
	print("2. Global fun csv not present")
	err[1] = 1

try:
	fi[2] = open("stats/" + af + "classes.csv")
except:
	print("3. classes csv not present")
	err[2] = 1

try:
	fi[3] = open("stats/" + af + "params.csv")
except:
	print("4. params csv not present")
	err[3] = 1

try:
	fi[4] = open("stats/" + af + "decls.csv")
except:
	print("5. decls csv not present")
	err[4] = 1
#f6 = open(af+"stats/fields.csv")

try:
	fi[5] = open("stats/" + af + "inheritance.csv")
except:
	print("6. inheritance csv not present")
	err[5] = 1

file_pointers = []
for i in range(6):
	if err[i] == 0:
		file_pointers.append(fi[i])

with open("stats/" + af + "methods.txt", "rb") as fp1:   # Unpickling
	methods = pickle.load(fp1)

with open("stats/" + af + "param.txt", 'rb') as fp2:
	param = pickle.loads(fp2.read())

with open("stats/" + af + "decl_var.txt", 'rb') as fp3:
	decl_var = pickle.loads(fp3.read())

with open("stats/" + af + "fields.txt", 'rb') as fp4:
	fields = pickle.loads(fp4.read())

# print(methods)
# print(param)
# print(decl_var)
# print(fields)



token_types = ['global_variable', 'global_func','classes', 'param_var', 'declared_var','class_fields']
nodeid = 1
for i in range(len(file_pointers)):
	for line in file_pointers[i]:
		arr = line.split(',')
		if i!=1:
			tup =  (nodeid, arr[0],get_styles(arr[0]),token_types[i],arr[3],arr[4])
		if i==1:
			tup =  (nodeid, arr[0][:-11],get_styles(arr[0]),token_types[i],arr[3],arr[4])
		token_nodes.append(tup)
		all_nodes[nodeid] = tup
		nodeid += 1
	file_pointers[i].seek(0)

# print(token_nodes)

concept_to_id = {}

modifiers = ['global', 'extern', 'inline', 'overload', 'virtual', 'public', 'private', 'static', 'const']
for modifier in modifiers:
	tup = (nodeid, modifier, 'modifier')
	concept_nodes.append(tup)
	all_nodes[nodeid] =  tup
	concept_to_id[modifier] = nodeid
	nodeid +=1

for domain_term in program_domain:
	tup = (nodeid, domain_term, 'program_domain')
	concept_nodes.append(tup)
	all_nodes[nodeid] =  tup
	concept_to_id[domain_term.lower()] = nodeid
	nodeid +=1

for domain_term in problem_domain:
	tup = (nodeid, domain_term, 'problem_domain')
	concept_nodes.append(tup)
	all_nodes[nodeid] =  tup
	concept_to_id[domain_term.lower()] = nodeid
	nodeid +=1

if "updates" in concept_to_id:
	print "Yes, working well"
# print(concept_nodes)
#traversing token_nodes in the same order again
nodeid=1
identifier_tokens = []
for i in range(len(file_pointers)):
	#print(i)
	for line in file_pointers[i]:
		arr = line.split(',')
		#if i==0 or i==1:
			#relations.append((nodeid,concept_to_id['global'],1))
			#print(relations)
		#if arr[5] == 'public':
			#relations.append((nodeid,concept_to_id['public'],1))
		#elif arr[5] == 'private':
			#relations.append((nodeid,concept_to_id['private'],1))
		
		#if arr[6] == 1:
			#relations.append((nodeid,concept_to_id['overload'],1))

		#if arr[7] == 1:
			#relations.append((nodeid,concept_to_id['isVirtual'],1))

		#if arr[8] == 1:
			#relations.append((nodeid,concept_to_id['isStatic'],1))

		#if arr[9] == 1:
			#relations.append((nodeid,concept_to_id['isInline'],1))

		#if arr[10] == 1:
			#relations.append((nodeid,concept_to_id['isExtern'],1))

		#if arr[11] == 1:
			#relations.append((nodeid,concept_to_id['isConstant'],1))
		print ""
		print "Text: ", arr[0]
		identifier_tokens.append([arr[0], all_nodes[nodeid][4], all_nodes[nodeid][5]])
		prog_toks = prog_tokens(arr[0])
		print("Prog toks:", prog_toks)
		if len(prog_toks) > 0:
			for tok in prog_toks:
				if tok in concept_to_id:
					#print('present')
					relations.append((all_nodes[nodeid][1], all_nodes[nodeid][4], all_nodes[nodeid][5], concept_to_id[tok.lower()], tok, 1))


		#print('parsing ' + arr[0])
		print ""
		prob_toks = prob_tokens(arr[0])
		print("Prob Toks:", prob_toks)
		if len(prob_toks) > 0:
			for tok in prob_toks:
				if tok in concept_to_id:
					#print('present')
					relations.append((all_nodes[nodeid][1], all_nodes[nodeid][4], all_nodes[nodeid][5], concept_to_id[tok.lower()], tok, 2))
		#else:
			#print('absent')
		nodeid += 1

print(relations)
# # accSpec, isOverload, isVirtual, isStatic, isInline, isExtern, isConstant
# # 5			6			7			8		9		10 			11

out = [relations, identifier_tokens]

import pickle
with open("identifier_graph.p", 'wb') as fp:
	pickle.dump(out, fp, protocol=2)

#print all_nodes[1]

