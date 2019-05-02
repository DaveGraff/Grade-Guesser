#For tagger
from sklearn.linear_model import LogisticRegression
import pickle

#For Data
import json

def getTextEmbeddings():
	#Find adjectives with tagger
	#Make a text embedding for each
	#Average them all
	return

def makeFeatureVector():
	#getTextEmbeddings()
	#add together w/ other features in 1D arr
	return


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
	#Create feature vectors


	#Plug and chug data in RNN