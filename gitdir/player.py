import shelve
from os import system
d=shelve.open("data")["train"]

for files in d:
	print d
#	system("play %s" %files[0])
#	system("play %s" %files[1])
