import MySQLdb
import os
from glob import glob
import csv
from contextlib import closing
import sys

PATH = '/home/srijoni/Desktop/LLVM_Install/llvm/build/input_files/symengine/symengine'
cc_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.cc'))]
cpp_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.cpp'))]
c_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.c'))]
h_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.h'))]

files = c_files + cc_files + cpp_files

print(files)