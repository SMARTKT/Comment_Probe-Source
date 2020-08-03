#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  StanfordParsing.py for comment analysis of c/c++ programs using AST and POS and dependencies
#  
#  Copyright 2018 Shakti <shakti@shakti-Inspiron-5559>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  



import os
import re
import nltk
nltk.data.path.append('/home/srijoni/nltk_data')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize, sent_tokenize

from nltk.parse.stanford import StanfordParser
from nltk.tree import ParentedTree, Tree
from nltk import tokenize
import graphviz
from nltk.parse.stanford import StanfordDependencyParser
import pickle 

from nltk import pos_tag


import MySQLdb

import re

def sentence_splitter(sentence) :
	re.findall(r"[\w']+|[.,!?;]", sentence)

#def words_tok(text) :
#	return re.findall("[\O\Θ\Ωa-zA-Z0-9'!#$%&*+-_/=?`{|}~.\"\(\):;.,]+", text, re.UNICODE)

informativeCommentCheckList = ['copyright','warranty','merchantability','public license', 'program for', 'application for' , 'code for', 'prg for' , 'C code', 'C program']

db = MySQLdb.connect("localhost","root","srijoni321","Comments" ,charset='utf8',
                     use_unicode=True )

# prepare a cursor object using cursor() method
cursor = db.cursor()

 
 
print("os environment")
path_to_jar = '../StanFordParser/stanford-parser-full-2015-12-09/stanford-parser.jar' 
path_to_models_jar = '../StanFordParser/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar'

os.environ['STANFORD_PARSER'] = path_to_jar
os.environ['STANFORD_MODELS'] = path_to_models_jar
Words = dict()

def find_subject(t):
  for s in t.subtrees(lambda t: t.label() == 'NP'):
    for n in s.subtrees(lambda n: n.label().startswith('NN')):
      return (n[0], find_attrs(n))
      


def find_predicate(t):
  v = None
 
  for s in t.subtrees(lambda t: t.label() == 'VP'):
    for n in s.subtrees(lambda n: n.label().startswith('VB')):
      v = n
  return (v[0], find_attrs(v))
  

def find_object(t):
  for s in t.subtrees(lambda t: t.label() == 'VP'):
    for n in s.subtrees(lambda n: n.label() in ['NP', 'PP', 'ADJP']):
      if n.label() in ['NP', 'PP']:
        for c in n.subtrees(lambda c: c.label().startswith('NN')):
          return (c[0], find_attrs(c))
      else:
        for c in n.subtrees(lambda c: c.label().startswith('JJ')):
          return (c[0], find_attrs(c))
          
          
          
def find_attrs(node):
  attrs = []
  p = node.parent()
 
  # Search siblings of adjective for adverbs
  if node.label().startswith('JJ'):
    for s in p:
      if s.label() == 'RB':
        attrs.append(s[0])
 
  elif node.label().startswith('NN'):
    for s in p:
      if s.label() in ['DT','PRP$','POS','JJ','CD','ADJP','QP','NP']:
        attrs.append(s[0])
 
  # Search siblings of verbs for adverb phrase
  elif node.label().startswith('VB'):
    for s in p:
      if s.label() == 'ADVP':
        attrs.append(' '.join(s.flatten()))
 
  # Search uncles
  # if the node is noun or adjective search for prepositional phrase
  if node.label().startswith('JJ') or node.label().startswith('NN'):
      for s in p.parent():
        if s != p and s.label() == 'PP':
          attrs.append(' '.join(s.flatten()))
 
  elif node.label().startswith('VB'):
    for s in p.parent():
      if s != p and s.label().startswith('VB'):
        attrs.append(' '.join(s.flatten()))
 
  return attrs                  




def check(sent) :
	
	parser = StanfordParser()
 
	# Parse the example sentence
	
	
	print(sent)
	t = list(parser.raw_parse(sent))[0]
	print(t)
	t = ParentedTree.convert(t)
	print(t)
	t.pretty_print()
	try :
		subj = find_subject(t)
	except :
		subj = []
	try :
		pred = find_predicate(t)
	except :
		pred = []
	try :
		obj =  find_object(t)
	except :
		obj = []
		
	print (subj)
	print (pred)
	print (obj)
	return subj , pred , obj
	
def findDependencies(sentence) :
	global Words
	try :
		dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
		result = dependency_parser.raw_parse(sentence)
	except :
		print("Error in parsing the tree")
	
	
	Words = dict()
	
	
	#parsetree = list(result)[0]
	#print("Parse Tree")
	#print(parsetree)
	
	root_tag = []
	roots = []
	pos_tagging = []
	
	dependencyList= []
	maxIndex = -1
	try :
		parsetree = list(result)[0]
		for k in parsetree.nodes.values():
			if k["ctag"] != None and k["word"] != None:
				w = k['word']
				address = int(k['address']) - 1
				pos_tagging.append([address,w,k['ctag']])
				Words[address] = w
				if maxIndex < address :
					maxIndex = address
	except :
		print("Error in creating pos tagging")
		pos_tagging = []
		return pos_tagging , roots , dependencyList , Words
	
	for i in range(0, maxIndex + 1) :
		try :
			x = Words[i]
		except :
			Words[i] = ""
	print("-------------- Dictionary Of Words ----------------------------")	
	print(Words)
	
	try :
		dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
		result = dependency_parser.raw_parse(sentence)
	except :
		print("Error in parsing the tree")
			
	try:
		parsetree = list(result)[0]
		for k in parsetree.nodes.values():
			
			if k["head"] == 0:
				root_tag.append(str(k["rel"])  + "(" + "Root" + "-" + str(k["head"]) + "," + str(k["word"]) + "-" + str(k["address"]) + ")" )
				rootAddress = k["address"] - 1
				leftWords = []
				rightWords = []
				for i in range(0, rootAddress) :
					leftWords.append(Words[i])
				for i in range(rootAddress + 1 , maxIndex + 1) :
					rightWords.append(Words[i])
					
				
					
				roots.append([rootAddress,k["word"], leftWords , rightWords])
				print(k)
			else:
				print(k)
			if k["ctag"] != None and k["word"] != None:
				w = k['word']
				address = int(k['address']) - 1
				t1 = (w , address)
				#print("***** Deps for word " + w)
				cdepts = k['deps']
				print("****************Test1******************")
				print(cdepts)
				for key in cdepts :
					print("Key = ")
					#print(key)
					dependencies = cdepts[key]
					print("******************Test2***************")
					for dep in dependencies :
						print("Dep  =")
						#print(dep)
						
						print("Words at dep = ")
						print(Words[int(dep)-1])
						addr = int(dep) - 1
						t2 =(Words[addr] , addr )
						dependencyList.append([key , t1 , t2])
					print("*****************************************")
					
	except :
		    print("Error in reading nodes of the tree")			
	print(root_tag)
	print(roots)
	print("POS-Tagging")
	print(pos_tagging)
	'''
	dependencyList = []
	try :
		res = dependency_parser.raw_parse(sentence)
		dep = res.__next__()
		depList = list(dep.triples())
		
		for x in depList :
			print(x[0])
			print(x[1])
			print(x[2])
			print('------------------')
			dependencyList.append([x[0],x[1],x[2]])
	except :
		print("Error in finding dependencies")
		
	print("____________________________________________________________________")
	'''
	print(dependencyList) 
	i = 0
	while i < len(Words) :
		try :
			x = Words[i]
		except :
			Words[i] = ""
		i = i + 1
	
	   
	    
	return pos_tagging , roots , dependencyList, Words  


sentence = raw_input()

print "++++ Split sentencee"
print sentence_splitter(sentence)

print "++++ Check"
print check(sentence)

print "++++ Find dependencies"
pos_tagging , roots , dependencyList, Words = findDependencies(sentence)
print "Postagging:", pos_tagging
print "Roots:", roots
print "DependencyList:",dependencyList
print "Words:", Words
# print "++++ Subj: ", find_subject(sentence)
# print "++++ Obj: ", find_object(sentence)
# print "++++ Pred: ", find_predicate(sentence)
