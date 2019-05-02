import pickle
import os
import course
import json

addedCount = 0
fileNum = 0
#addFile = open('filterData.bin', 'wb')
data = []

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
