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
import numpy as np
import tensorflow as tf
import argparse
	

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
	features.append(['LEC', 'SEM', 'LAB', 'REC', 'TUT', 'CLN'].index(course['courseType'].replace('\r', '')))
	#Add professor Num
	features.append(len(course['professors'].split(r'(and)|,')))
	return features


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
		if not len(word) == 0 and tagger.predict([getFeaturesForTarget(wordsArr, idx)]) == [16]:
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


def calculateAllReviewVectorData(data):
	print("Calculating review data")
	a1 = np.array(getReviewVector(data[0]))
	a2 = np.array(getReviewVector(data[1]))
	reviewVectorData = (np.vstack((a1, a2)))

	#Plug and chug data in RNN
	for i in range( 2, len(data)):
		if i%15==0:
			print("Progress: " + str(i) + "/" + str(len(data)))
		currentData = np.array(getReviewVector(data[i]))
		reviewVectorData = np.vstack((reviewVectorData, currentData))

	print("Review vector data calculated with shape " + str(reviewVectorData.shape))
	print("Writing review vector data to file: reviewVectorData.csv")
	np.savetxt("reviewVectorData.csv", reviewVectorData, delimiter=",")
	return reviewVectorData

def calculateAllMetaVectorData(data):
	print("Calculating Meta data")
	a1 = np.array(getMetaVector(data[0]))
	a2 = np.array(getMetaVector(data[1]))
	metaVectorData = (np.vstack((a1, a2)))

	#Plug and chug data in RNN
	for i in range(2, len(data)):
		if i%15==0:
			print("Progress: " + str(i) + "/" + str(len(data)))
		currentData = np.array(getMetaVector(data[i]))
		metaVectorData = np.vstack((metaVectorData, currentData))

	print("Meta vector data calculated with shape " + str(metaVectorData.shape))
	print("Writing Meta vector data to file: metaVectorData.csv")
	np.savetxt("metaVectorData.csv", metaVectorData, delimiter=",")
	return metaVectorData


if __name__ == "__main__":	
	parser = argparse.ArgumentParser()
	parser.add_argument("--calc", action='store_true', help="calcualte and write reviewVectorData")
	args = parser.parse_args()

	f = open("courses.txt")
	codeList = f.read().splitlines()
	f.close()

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
	
	f = open("Word2Vec/CBOW.bin", 'rb')
	embedding = pickle.load(f)
	f.close()
	print("Vector Embedding loaded")

	#Load data
	f = open("data.json")
	data = json.load(f) #Array of dicts
	f.close()
	print("Data loaded")

	if args.calc:
		metaVectorData = calculateAllMetaVectorData(data)
	else:
		print("reading metaVectorData form file metaVectorData.csv")
		reviewVectorData = np.genfromtxt("metaVectorData.csv", delimiter=",")