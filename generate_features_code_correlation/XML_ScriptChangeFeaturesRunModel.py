import os


os.system("python2 XML_BuildCatForBunch.py excels1.txt")
os.system("python2 XML_BuildCatForBunch.py excels2.txt")
os.system("python2 XML_BuildCatForBunch.py excels3.txt")
os.system("python2 XML_BuildCatForBunch.py excels4.txt")

os.system("python2 XML_ClubFiles.py files1.txt files1.csv")
os.system("python2 XML_ClubFiles.py files2.txt files2.csv")
os.system("python2 XML_ClubFiles.py files3.txt files3.csv")
os.system("python2 XML_ClubFiles.py files4.txt files4.csv")

os.system("python2 XML_merge4CSV.py files1.csv files2.csv files3.csv files4.csv 389 399 150 121")
print "Done fetaures extraction"

os.system("python2 Models/check.py Models/training.csv Models/Y.csv")
print "Model generated"
