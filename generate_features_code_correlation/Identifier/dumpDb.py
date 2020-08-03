import os

count = 1
start = x
with open('locations.txt') as f:
	for line in f:
		if count < start:
			count += 1
			continue
		arr = line.split()
		cmd = '../../../home/srijoni/spandan/llvm_build_copy/build/bin/syntax-tree ' + arr[1]
		os.system(cmd)
		cmd2 = 'mysqldump -u root -p srijoni321 test  > ' + arr[0] + '.db'
		os.system(cmd2)
		count += 1
