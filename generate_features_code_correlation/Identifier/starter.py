import MySQLdb
import os
from glob import glob
import csv

connection = MySQLdb.connect(
                host = 'localhost',
                user = 'root',
                passwd = 'srijoni321')  # create the connection

cursor = connection.cursor()     # get the cursor

cursor.execute("SHOW databases;")
dbs = cursor.fetchall()

arr = []
for db in dbs:
	arr.append(db[0])

if "test" in arr:
	print("test already exists")
	cursor.execute("USE test;") # select the database
else:
	print("created new database test")
	cursor.execute("CREATE database test;")
	cursor.execute("USE test;")

