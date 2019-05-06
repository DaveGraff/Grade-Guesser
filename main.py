#For tagger
from sklearn.linear_model import LogisticRegression
import pickle
#For Data
import json
import re

f = open("courses.txt")
codeList = f.read().splitlines()
f.close()
	

def getMetaVector(course):
	features = []
	#Add number of students
	features.append(course['studentNum'])
	#Add 3 digit course code
	features.append(int(course['code'][3:]))
	#Add department
	features.append(codeList.index(course['code'][:3].lower()))
	#Add semester
	features.append(['Winter', 'Spring', 'Summer', 'Fall'].index(course['semester'].split(" ")[0]))
	#Add year
	features.append(int(course['semester'].split(" ")[1]) - 2014)
	#Add course type
	features.append(['LEC', 'SEM', 'LAB', 'REC'].index(course['courseType'].replace('\r', '')))
	#Add professor Num
	features.append(len(course['professors'].split(r'[and]|,')))
	return features



if __name__ == "__main__":
	
	#Load tagger
	f = open("Tagger/tagger.bin", 'rb')
	tagger = pickle.load(f)
	f.close()
	print("Tagger loaded")

	#Load data
	f = open("data.json")
	data = json.load(f); #Array of dicts
	f.close()
	print("Data loaded")
	print(data[100]['courseType'])
	print(getMetaVector(data[0]))
	#Create feature vectors


	#Plug and chug data in RNN