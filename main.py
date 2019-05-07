# This file puts all the others together and computes
# the final result. It takes the trained tagger, the
# vector embedding, and the filtered data in .bin and
# JSON files. Unfortunately, some of the tagger methods
# to be copied over to save a bit of time instead of
# figuring out imports. This page also constructs the 
# featureVector for both the meta data and review data,
# and feeds them into an RNN as necessary.

#For tagger
from sklearn.linear_model import LogisticRegression
import pickle
#For Data
import json
import re

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 

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
	features.append(len(course['professors'].split(r'(and)|,')))
	return features


#Load gazeteer & wordToIndex
f = open('Tagger/gazeteer.txt', 'r')
gazeteer = [line.replace('\n', '') for line in f.readlines()]
f.close()
f = open("wordToIndex.bin", 'rb')
wordToIndex = pickle.load(f)
f.close()
print("Loaded gazeteer & wordToIndex")

#Load tagger
f = open("Tagger/tagger.bin", 'rb')
tagger = pickle.load(f)
f.close()
print("Tagger loaded")

def getWordIndex(word, wordToIndex):
    returnList = []
    i = 0
    if word.lower() in wordToIndex:
        while i < len(wordToIndex) and i < wordToIndex[word.lower()]:
            returnList += [0]
            i += 1
        returnList += [1]
        i += 1
        while i < len(wordToIndex):
            returnList += [0]
            i += 1
    else:
        while i < len(wordToIndex):
            returnList += [0]
            i += 1
    return returnList

def isInGazeteer(tokens, targetI, gazeteer):
    if targetI < 0 or targetI == len(tokens):
        return [0]
    if tokens[targetI] in gazeteer:
        return [1]
    return [0]

def getFeaturesForTarget(tokens, targetI):
    featureVector = []
    target_word = tokens[targetI]
    letterOrd = ord(target_word[0])
    #Figure out if next, previous, or current word is common pn
    featureVector += isInGazeteer(tokens, targetI-1, gazeteer)
    featureVector += isInGazeteer(tokens, targetI, gazeteer)
    featureVector += isInGazeteer(tokens, targetI+1, gazeteer)
    #Is it the first word?
    if not targetI == 0:
        if tokens[targetI-1] == '':
            featureVector += [0]
        else:
            featureVector += [0]
    else:
        featureVector += [0]
    #Is capitalized?
    if letterOrd > 64 and letterOrd < 91:
        featureVector += [1]
    else:
        featureVector += [0]
    #257 1 hot values for 1st letter (last for non-ASCII)
    i = 0
    while i < letterOrd and i < 256:
        featureVector += [0]
        i += 1
    featureVector += [1]
    i += 1
    if letterOrd < 256:
        while i < 257:
            featureVector += [0]
            i += 1
    #Word length
    featureVector.append(len(target_word))

    #One-hot of previous word?
    i = 0
    if targetI == 0:# No previous word
        while i < len(wordToIndex):
            featureVector += [0]
            i += 1
    else:
        featureVector += getWordIndex(tokens[targetI-1], wordToIndex)

    #One-hot of current word??
    featureVector += getWordIndex(tokens[targetI], wordToIndex)

    #One-hot of next word???
    i = 0
    if targetI ==len(tokens) -1:# No next word
        while i < len(wordToIndex):
            featureVector += [0]
            i += 1
    else:
        featureVector += getWordIndex(tokens[targetI+1], wordToIndex)

    return featureVector

f = open("Word2Vec/CBOW-Election.bin", 'rb')
embedding = pickle.load(f)
f.close()
print("Vector Embedding loaded")

def meetsThreshold(control, test):
	num = 0
	try:
		num = embedding.wv.similarity(control, test)
	except:
		return 0
	if num >= .45:
		return 1
	return 0

def getReviewVector(course):
	features = []
	#Add number of reviews
	features.append(len(course['improvedComm']) + len(course['valuableComm']))
	#Add total size of reviews
	total = 0
	#Combine text for ease of processing
	text = ''
	for comment in course['improvedComm']:
		total += len(comment.split(' '))
		text += comment
	for comment in course['valuableComm']:
		total += len(comment.split(' '))
		text += comment
	features.append(total)
	#Add hours spent studying & attendance
	features.extend(course['studying'])
	features.extend(course['attendance'])
	#Define words to search for
	profMentions = ['professor', 'professors', 'prof', 'teacher', 'lecturer', 'ta']
	for name in course['professors'].replace(r'(and )|,|\r', '').split(' '):
		profMentions.append(name)
	#Find prof mentions
	wordsArr = text.split(' ')
	total = 0
	for word in wordsArr:
		if word.lower() in profMentions:
			total += 1
	features.append(total)

	#Find adjectives
	adj = []
	idx = 0
	for word in wordsArr:
		if tagger.predict([getFeaturesForTarget(wordsArr, idx)]) == [16]:
			adj.append(word.lower())
		
		idx += 1
	
	#Counters for similarity
	hard = 0
	easy = 0
	fun = 0
	boring = 0
	
	for word in adj:
		hard += meetsThreshold('hard', word)
		easy += meetsThreshold('easy', word)
		fun += meetsThreshold('fun', word)
		boring += meetsThreshold('boring', word)

	features.extend([hard, easy, fun, boring])

	return features



if __name__ == "__main__":
	

	#Load data
	f = open("data.json")
	data = json.load(f); #Array of dicts
	f.close()
	print("Data loaded")
	print(getReviewVector(data[0]))


	#Plug and chug data in RNN