#!/usr/bin/env python
# coding: utf-8

# This is a tagger similar to the one used for assignment 1.
# It's necessary for the feature vector as described in our
# proposal. Due to severe underfitting of the data, we found it
# quicker and easier to keep all elements, instead of training it
# to recognize only Adjectives.

import re
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sys import argv
import pickle

def tokenize(sent):
    p = re.compile("[#@]?(?:[A-Z]\.|[a-z]|[A-Z]|')+|\.\.\.|[\.\?,!:;\"'\(\)\[\]]")
    return p.findall(sent)

def isInGazeteer(tokens, targetI, gazeteer):
    if targetI < 0 or targetI == len(tokens):
        return [0]
    if tokens[targetI] in gazeteer:
        return [1]
    return [0]

def getFeaturesForTarget(tokens, targetI, wordToIndex, gazeteer):
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

#Helper function for Part 2
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


def trainTagger(features, tags):
    penaltyC = 10
    penaltyType = 'l2'
    model = LogisticRegression(C=penaltyC, penalty=penaltyType, random_state=25, solver="liblinear", multi_class="auto")
    model.fit(features, tags)
    return model


def testAndPrintAcurracies(tagger, features, true_tags):
    pred_tags = tagger.predict(features)
    print("\nModel Accuracy: %.3f" % metrics.accuracy_score(true_tags, pred_tags))
    for tag in pred_tags:
        if tag == 16:
            print(tag)
    #most Frequent Tag: 
    mfTags = [Counter(true_tags).most_common(1)[0][0]]*len(true_tags) 
    print("MostFreqTag Accuracy: %.3f" % metrics.accuracy_score(true_tags, mfTags))
    

#input: filename for a conll style parts of speech tagged file
#output: a list of list of tuples [sent]. representing [[[word1, tag], [word2, tag2]]
def getConllTags(filename):    
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f: 
            wordtag=wordtag.strip()
            if len(wordtag) == 1:
                continue
            if wordtag:#still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word,tag))
            else:#new sentence
                wordTagsPerSent.append([])
                sentNum+=1
    return wordTagsPerSent  

corpus1 = 'daily547.conll'
corpus2 = 'oct27.conll'

def taggerTrainer():
    #2) Run Feature Extraction:
    #2a) load training data: 
    wordToIndex = set()
    tagToNum = set()
    taggedSents = getConllTags(corpus1)
    taggedSents += getConllTags(corpus2)
    for sent in taggedSents:
        if sent: 
            words, tags = zip(*sent)
            wordToIndex |= set([i for i in words if words.count(i) > 1]) #union of the words into the set
            tagToNum |= set(tags) #union of all the tags into the set
    print("[Read ", len(taggedSents), " Sentences]")

    #make dictionaries for converting words to index and tags to ids:
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)} 
    numToTag = list(tagToNum) #mapping index to tag
    tagToNum = {numToTag[i]: i for i in range(len(numToTag))}
    
    #load gazeteer
    gazeteerFile = open('gazeteer.txt', 'r')
    gazeteer = [line.replace('\n', '') for line in gazeteerFile.readlines()]
    #2b) Call feature extraction on each target
    X = []
    y = []
    for tag in tagToNum:
        if tag == 'A':
            print(tagToNum[tag])
        #     tagToNum[tag] = 1
        # else:
        #     tagToNum[tag] = 0

    print("[Extracting Features]")
    for sent in taggedSents:
        if sent: 
            words, tags = zip(*sent)
            for i in range(len(words)):
                y.append(tagToNum[tags[i]]) #append y with class label
                X.append(getFeaturesForTarget(words, i, wordToIndex, gazeteer))
    X, y = np.array(X), np.array(y)
    print("[Done X is ", X.shape, " y is ", y.shape, "]")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    print("[Broke into training/test. X_train is ", X_train.shape, "]")
    
    ####################################################
    #3 Train the model. 
    print("[Training the model]")
    tagger = trainTagger(X_train, y_train)
    print("[done]")
    testAndPrintAcurracies(tagger, X_test, y_test)
    return tagger

if __name__ == "__main__":
    tagger = taggerTrainer()
    if(len(argv) == 3 and argv[1] == '-s'):
        print("Saving tagger in provided file")
        file = open(argv[2], 'wb')
        pickle.dump(tagger, file)
        file.close()
   
