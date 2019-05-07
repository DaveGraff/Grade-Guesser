# This iterates though the raw HTML files downloaded from
# Classie-Evals and sends them to be parsed into Course objects.
# It then stores arrays of those objects into a series of binary
# files.

import course
import pickle
import os

fileName = 'courseData'
fileType = '.bin'
batchSize = 100

i = 0
loop = 1
f = open(fileName + '0' + fileType, 'wb')
for file in os.listdir("Raw Data"):
	tempClass = course.Course("Raw Data/" + file)
	pickle.dump(tempClass, f)
	if i == batchSize:
		print('Completed batch', loop)
		i = 0
		if loop % 50 == 0:
			f.close()
			f = open(fileName + str(int(loop/30)) + fileType, 'wb')
			print('Starting new file')
		loop += 1
	i += 1
f.close()
