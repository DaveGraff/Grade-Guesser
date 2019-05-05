import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import Word2Vec 
import json
import pickle
  
f = open("../data.json", "r") 
data = json.load(f)
f.close()

# Replaces escape character with space 
# f = s.replace(r"[\r\n]", " ") 
  
newData = [] 
index = 0

for course in data:
	relev = course['valuableComm'] + course['improvedComm']
	for comment in relev:
		for i in sent_tokenize(comment):
			temp =[]

			for j in word_tokenize(i.replace(r"[\r\n]", " ").lstrip()):
				temp.append(j.lower())

			newData.append(temp)
	index += 1
	if index % 50 == 0:
		print("Finished batch", index/50)


print("Training models")
  
# # Create CBOW model 
model1 = gensim.models.Word2Vec(newData, min_count = 1, size = 100, window = 5) 
f = open("CBOW.bin", 'wb')
pickle.dump(model1, f)
f.close


  
# Create Skip Gram model 
model2 = gensim.models.Word2Vec(newData, min_count = 1, size = 100, window = 5, sg = 1) 
f = open('skipgram.bin', 'wb')
pickle.dump(model2, f)
f.close()

print("Models are saved")