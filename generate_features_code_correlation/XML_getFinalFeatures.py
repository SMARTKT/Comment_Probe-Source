import csv
import sys
import os
import re
import os
import pickle
import editdistance
from NLP_features_feb import NLP_features


# Text Categories code

CATEGORY_COUNT = 30
CAT_CONCEPTS_MATCH_SYMBOLS = 1
CAT_CONCEPTS_MATCH_TYPE = 2
CAT_CONCEPTS_NOT_MATCH_SYMBOLS = 3
CAT_CONCEPTS_PARTIALLY_MATCH_SYMBOLS = 4
CAT_CONCEPTS_MATCH_STRUCTURE = 5
CAT_NO_PROGRAM_DOMAIN_CONCEPTS = 6
CAT_NO_PROBLEM_DOMAIN_CONCEPTS = 7
CAT_LOW_PROGRAM_DOMAIN_CONCEPTS = 8
CAT_HIGH_PROGRAM_DOMAIN_CONCEPTS = 9
CAT_LOW_PROBLEM_DOMAIN_CONCEPTS = 10
CAT_HIGH_PROBLEM_DOMAIN_CONCEPTS = 11
CAT_CODE_COMMENT = 12
CAT_SHORT = 13
CAT_HIGH_SCOPE = 14
CAT_LOW_SCOPE = 15
CAT_COPYRIGHT_LICENSE = 16
CAT_DATE = 17
CAT_EMAIL = 18
CAT_CONTACT = 19
CAT_BUG_VERSION = 20
CAT_AUTHOR_NAME = 21
CAT_BUILD = 22
CAT_EXCEPTION = 23
CAT_PERFORMANCE = 24
CAT_DESIGN = 25
CAT_MEMORY = 26
CAT_SYSTEM_SPEC = 27
CAT_LIBRARY = 28
CAT_OUTPUT_RETURN = 29
CAT_JUNK = 30

def is_verb(tag):
	return tag in ['VB', 'VBZ', 'VBN', 'VBP', 'VBG', 'VBD']

def is_conditional(postags, dependencies):
	for postag in postags:
		if postag[1]=='IN':
			for dependency in dependencies:
				if dependency[1] == 'mark' and (dependency[0][1] == 'IN' or dependency[2][1] == 'IN'):
					return True
	return False

# use stanford parser to categories comments based on postags and dependencies
# list of categories:
# CONDITIONAL: contains 'IN' tag and has a 'mark' dependency involving the 'IN' tag
# NN_JJ_SYM_ROOT: there is a NN, JJ or SYM tag as ROOT
# VERB_ROOT:  there is a verb (VB, VBZ, VBN, VBP, VBG, VBD) tag as ROOT
# VERB_AUXILIARY: the ROOT is not verb, but there is an auxiliary verb
#
def find_nlp_categories(comment):
	result = []

	postags = stanford_pos_tagger.tag(comment.split())
	raw_dependencies = [parse for parse in stanford_dep_parser.raw_parse(comment)]
	dependencies = [list(dep.triples()) for dep in raw_dependencies]

	#for now, assume a single dependency tree
	raw_dependencies = raw_dependencies[0]
	dependencies = dependencies[0]

	#check for each category

	if is_conditional(postags, dependencies):
		result.append('CONDITIONAL')

	if raw_dependencies.root['tag'] in ['NN', 'JJ', 'SYM']:
		result.append('NN_JJ_SYM_ROOT')

	if is_verb(raw_dependencies.root['tag']):
		result.append('VERB_ROOT')
	else:
		for postag in postags:
			if is_verb(postag[1]):
				result.append('VERB_AUXILIARY')
				break

	return result

def matches_with_keywords(text, keywords):
	text = text.lower()
	matches = 0
	for keyword in keywords:
		if keyword in text:
			matches = matches + 1

	return matches

def is_copyright_or_license_comment(comment):

	keywords = [
				"copyright", "copyleft", "copy-right", "license", "licence", "trademark", "open source", "open-source"
				]
	return matches_with_keywords(comment, keywords)

def is_bug_or_version_related_comment(comment):

	keywords = [
				" bug", "bug #", "bugid", "bug id", "bug number", "bug no", "bugno", "bugzilla",    # debug should not match
				" fix", "fix #", "fixid", "fix id", "fix number", "fix no", "fixno",   				# postfix, suffix etc should not match
				"patch", "patch #", "patchid", "patch id", "patch number", "patch no", "patchno",
				]

	ans = matches_with_keywords(comment, keywords)
	if is_copyright_or_license_comment(comment) == 0:
		ans += len(re.findall("bug [0-9]|fix [0-9]|version [0-9]", comment))
	return ans


def is_build_related_comment(comment):

	keywords = [
				"cmake", "makefile", "build", "g++", "gcc", "dependencies", "apt-get", ".rules",
				"git clone", "debug", "bin/", "yum", "install", "path"
				]

	return matches_with_keywords(comment, keywords)

def is_system_spec_related_comment(comment):

	keywords = [
				"ubuntu", "endian", "gpu", "hyperthreading", "32-bit", "64-bit", "128-bit", "configuration", "specification"
				"32bit", "64bit", "128bit", "configure"
				]

	ans = matches_with_keywords(comment, keywords) + len(re.findall("[0-9] [gG][bB]|[0-9] [mM][bB]|[0-9] [kK][bB]|Windows", comment))
	return ans


def is_author_name_comment(comment):
	keywords = [
				"written by", "coded by", "developed by", "edited by", "modified by", "author", "contact",
				"fixed by", "contributed by"
				]
	return matches_with_keywords(comment, keywords)

def is_date_comment(comment):
	keywords = [
				"date", "edited on", "written on", "created on", "modified on"
				]

	return matches_with_keywords(comment, keywords) + len(re.findall("\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{1,2}[\-\/][a-zA-Z]{3}[\-\/]\d{2,4}", comment))

def is_email_comment(comment):
	keywords = [
				"mail dot com", "mail dot in", "email"
				]

	return matches_with_keywords(comment, keywords) + len(re.findall("([a-zA-Z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)", comment))


def is_todo_comment(comment):

	keywords = ["todo", "to-do"]
	return matches_with_keywords(comment, keywords)

def is_junk_comment(comment):
	# there are no letters or numbers in the comment
	if re.search("[a-zA-Z0-9]", comment) is None:
		return 1
	return 0


def is_high_scope_comment(count):
	if count > 3:
		return 1
	return 0

def has_library_comment(comment):
	if comment.find(".h") != -1:
		return 1
	return 0

def isZero(x):
	if x == 0:
		return 1
	return 0

def isGreater(x, y):
	if x > y:
		return 1
	return 0

def readCSV(fname):
	f = open(fname, 'r')
	r = csv.reader(f)
	out = []
	for each in r:
		while len(each) < 20:
			each.append("")
		out.append(each)
	f.close()
	return out


def main():
	if len(sys.argv) != 5:
		print "Give 3 arguments, location of excel_file, location of features file, location of xml file, location of code file"
		exit(-1)

	excel_file = sys.argv[1]
	features_file = sys.argv[2]
	xml_file = sys.argv[3]
	code_file = sys.argv[4]

	base = readCSV(excel_file)
	features = readCSV(features_file)

	output = []

	extractor = NLP_features(xml_file)

	for i in range(1, len(base)):

		comment_text = base[i][2]

		add = 0.05

		bug_or_version_related = is_bug_or_version_related_comment(comment_text) * 8
		build_related = is_build_related_comment(comment_text) * 10
		system_spec_related = is_system_spec_related_comment(comment_text) * 5
		build_details = bug_or_version_related + build_related + system_spec_related + add

		authorship_related = is_author_name_comment(comment_text) * 8
		email = is_email_comment(comment_text) * 10
		date = is_date_comment(comment_text) * 5

		author_details = authorship_related + email + date + add

		# todo = is_todo_comment(comment_text) &
		copyright_or_license = is_copyright_or_license_comment(comment_text) * 10
		junk = is_junk_comment(comment_text) * 10
		junk_copy = copyright_or_license + junk + add

		words = float(features[i][0]) * 0.7 + add
		prog_conc = float(features[i][1]) * 1.2 + add
		prob_conc = float(features[i][2]) * 1.8 + add
		descriptional = float(features[i][3]) * 0.8 + add
		conditional = float(features[i][5]) * 1.2 + add
		operational = float(features[i][4]) * 1.8 + add
		prog_domain_identifier_matches = float(features[i][11]) * 1.2 + add
		prob_domain_identifier_matches = float(features[i][12]) * 1.8 + add
		scope_score = float(features[i][7]) * 0.4 + float(features[i][8]) * 1.2 + float(features[i][9]) * 2.0 + add


		row = [base[i][0], base[i][2], words, prog_conc, prob_conc, descriptional, operational, conditional, prog_domain_identifier_matches, prob_domain_identifier_matches, scope_score]
		# row.extend([copyright_or_license, bug_or_version_related, build_related, system_spec_related, authorship_related, email, date, junk])
		row.extend([build_details, author_details, junk_copy])

		pos_tags = base[i][19]
		pos_tags = pos_tags.split("|")
		description_of_dataset = extractor.description_of_dataset(comment_text)
		working_summary = extractor.working_summary(comment_text, pos_tags)
		design_and_development = extractor.design_and_development(comment_text)
		exception = extractor.exception(comment_text)
		build_instr = extractor.build_instr(comment_text)
		commits_and_bugs = extractor.commits_and_bugs(comment_text)
		AST_symbols = extractor.AST_symbols(comment_text)
		comment_placement = extractor.comment_placement(comment_text)
		usage_of_imports = extractor.usage_of_imports(comment_text, code_file)
		row.extend([description_of_dataset, working_summary, design_and_development, exception, build_instr, commits_and_bugs, AST_symbols, comment_placement, usage_of_imports])
		output.append(row)

	ofname = features_file.replace("feature", "train")
	f = open(ofname, 'w')
	writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
	for each in output:
		writer.writerow(each)
	f.close()

main()
