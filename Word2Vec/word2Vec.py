# This file is responsible for creating a vector embedding
# models. Both the CBOW (Continuous Bag of Words) and skipgram 
# models were tested, and CBOW tended to perform better for our
# tasks. Due to the relatively modest range of opinions in our
# dataset, this model was trained on Twitter Data following the
# 2016 election instead. 

import csv
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import Word2Vec 
import pickle

data = []

with open('election_day_tweets.csv', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for idx, row in enumerate(csv_reader):
        for i in sent_tokenize(row[0].encode('utf8').decode()):
            temp = []

            for j in word_tokenize(i.replace("\n", " ")):
                temp.append(j.lower())

            data.append(temp)
        if idx % 50 == 0:
            print("Processed Batch", idx/50)



print("Training models")
  
# # Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5) 
f = open("CBOW.bin", 'wb')
pickle.dump(model1, f)
f.close


  
# Create Skip Gram model 
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1) 
f = open('skipgram.bin', 'wb')
pickle.dump(model2, f)
f.close()

print("Models are saved")