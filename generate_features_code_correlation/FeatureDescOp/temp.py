import os
import re
import nltk
nltk.data.path.append('/home/srijoni/nltk_data')

from nltk.parse.stanford import StanfordDependencyParser


path_to_jar = '../StanFordParser/stanford-parser-full-2015-12-09/stanford-parser.jar' 
path_to_models_jar = '../StanFordParser/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar'

os.environ['STANFORD_PARSER'] = path_to_jar
os.environ['STANFORD_MODELS'] = path_to_models_jar
dep_parser=StanfordDependencyParser(model_path= path_to_models_jar + "/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


print [parse.tree() for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")]
