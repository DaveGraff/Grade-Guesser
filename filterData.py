import pickle
import os
import course

addedCount = 0
fileNum = 0
addFile = open('filterData.bin', 'wb')

def filterData(course):
	if course.studentNum == 0 or course.avgGrade == 0:
		return
	try:
		if int(course.code[3:]) >= 500:
			return
	except ValueError: # Super weird courses
		return
	if len(course.improvedComm) == 0 and len(course.valuableComm) == 0:
		return

	#put in addFile
	global fileNum
	fileNum += 1

for file in os.listdir("Parsed Data"):
	f = open("Parsed Data/" + file, 'rb')
	stillFiles = True
	while stillFiles:
		try:
			filterData(pickle.load(f))
		except EOFError:
			stillFiles = False
	print("Finished file" + file)
	f.close()

print("Filtered 36403 courses to", fileNum)