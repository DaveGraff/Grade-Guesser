# This file is responsible for filtering through the Parsed
# files to find which we can use. For instance, some courses 
# only give grades in recitation. We also wanted to exclude
# grad courses or those with S/U grading. This left us with
# about ~15,000 courses at the end. The selected data would 
# then be put in a new binary file.

import pickle
import os
import course
import json

addedCount = 0
fileNum = 0
#addFile = open('filterData.bin', 'wb')
data = []
courseCode = []

def hasGPAGrade(gradesVector):
	grades = 0
	for i in range(11):
		grades+=gradesVector[i]
	return grades > 5

def filterData(course):
	if course.studentNum == 0 or course.avgGrade == 0:
		return
	try:
		if int(course.code[3:]) >= 500:
			return
	except ValueError: # Super weird courses
		return
	if len(course.improvedComm) + len(course.valuableComm) < 6:
		return

	if not hasGPAGrade(course.gradesVector):
		return
	aCourse = {
			 'code' : course.code,
			 'semester': course.semester,
			 'courseType': course.courseType,
			 'professors': course.professors,
			 'courseName': course.courseName,
			 'studentNum': course.studentNum,
			 'avgGrade': course.avgGrade,
			 'gradesVector': course.gradesVector,
			 'valuableComm': course.valuableComm,
			 'improvedComm': course.improvedComm,
			 'attendance': course.attendance,
			 'studying': course.studying
		}
	data.append(aCourse)
	k = str(aCourse['code'][3:]) + str(['Winter', 'Spring', 'Summer', 'Fall'].index(aCourse['semester'].split(" ")[0])) + str(['LEC', 'SEM', 'LAB', 'REC', 'TUT', 'CLN'].index(aCourse['courseType'].replace('\r', '')))
	courseCode.append(k)
	#put in addFile
	global fileNum
	fileNum += 1

for file in os.listdir("Parsed Data"):
	f = open("Parsed Data/" + file, 'rb')
	stillFiles = True
	while stillFiles:
		try:
			filterData(pickle.load(f))
		except:
			stillFiles = False
	print("Finished file" + file)
	f.close()

print("Filtered 36403 courses to", fileNum)
print("Writing " + str(len(data)) + " courses to data.json...")

with open('data.json', 'w') as outfile:
	json.dump(data, outfile, indent=4, sort_keys=True)

with open('courseCode.json', 'w') as outfile:
	json.dump(courseCode, outfile, indent=4, sort_keys=True)
