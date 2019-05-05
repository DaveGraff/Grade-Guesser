#For tagger
from sklearn.linear_model import LogisticRegression
import pickle

#For Data
import json
import re

def read_gazeteer(filename):
	"""
    Read a text file (list) and return a dict {'word': rank}
    :param filename:
    :return:
    """
	gazeteer = {}
	with open(filename, 'rt') as f:
		for r, w in enumerate(f.read().splitlines()):
			gazeteer[w] = r
	print("INFO: read gazeteer: %d entries from file %s" % (len(gazeteer.keys()), filename))
	return gazeteer

##################################### TAGGER SECTION #####################################################
def glue_tokens(tok):
	# INput: list of tokens
	# Output: list of token with certain combinations detoknized
	out = " ".join(tok)
	out = re.sub("n ' t ", "'t ", out)  # n't contraction
	out = re.sub(" ' s ", "'s ", out)  # 's contraction
	# handle abbreviations of up to three letters
	out = re.sub(r"([A-Z]) \. ([^$])", r'\1. \2', out)  # anything period within sentence will get glued back
	out = re.sub(r"([A-Z]) \. ([A-Z])", r'\1. \2', out)  # anything period within sentence will get glued back
	out = re.sub(r"\. ([A-Z])", r'.\1', out)  # anything period within sentence will get glued back
	out = re.sub(r"\. ([A-Z])", r'.\1', out)  # anything period within sentence will get glued back
	return out.split()

def tokenize(sent):
	# input: a single sentence as a string.
	# output: a list of each “word” in the text
	# must use regular expressions

	r = re.findall('([^\s,.!?\'\"-]+)|([,.!?\'\"-]+)', sent)
	r = [''.join(list(i)) for i in r]
	tokens = glue_tokens(r)
	return tokens

def one_hot_with_oov(word, w2i):
	# gets word and returns one_hot vector
	# accepts None and will return zero-only vector

	oh = np.zeros([len(w2i) + 1, ])
	if word is None or word not in w2i:
		oh[-1] = 1
		return oh
	idx = w2i[word]
	oh[idx] = 1
	return oh

def parseEntitiesFromTags(wordtag_seq):
    """
    Parses multi-word named entities based on wordtag_seq. Returns a list of strings tagged as adjectives from review.
    :param wordtag_seq:
    :return: list of strings
    """
    entity_list = []
    within_NE = False  # keeps track of being within or outside an entity
    entity = []
    for w, t in wordtag_seq:
        assert(t == 0 or t == 1)
        if within_NE:
            if t == 1:   # grow this entity just append
                entity.append(w)
            else:      # this is the end of the entity, append to output:
                within_NE = False
                entity_list.append(" ".join(entity))
                entity = []
        else:
            if t == 1:  # this is the beginning of an entity
                entity.append(w)
                within_NE = True
            else: # there is nothing here so just move on
                pass
    # finalize
    if within_NE:
        entity_list.append(" ".join(entity))
    # entity_list should have all occurrences of deisred entity
    return entity_list

def getTextEmbeddings():
	#Find adjectives with tagger
	#Make a text embedding for each
	#Average them all

	return

def makeFeatureVector(tokens, targetI, wordToIndex, gazeteer):
	#getTextEmbeddings()
	#add together w/ other features in 1D arr
	# input: tokens: a list of tokens,
	#       targetI: index for the target token
	#       wordToIndex: dict mapping ‘word’ to an index in the feature list.
	# output: list (or np.array) of k feature values for the given target

	prev_w = next_w = None
	curr_w = tokens[targetI]
	if targetI > 0:
		prev_w = tokens[targetI - 1]
	if targetI < len(tokens) - 1:
		next_w = tokens[targetI + 1]

	# feature 1 - is cap ?
	f1 = np.zeros([1, ], dtype=np.int)
	if curr_w[0].isupper():
		f1 = 1

	# feature 2: one-hot of first character
	f2 = np.zeros([257, ], dtype=np.int)
	o = ord(curr_w[0])
	if o <= 255:
		f2[o] = 1
	else:
		f2[-1] = 1

	# Feature 3: Length of curr w
	f3 = np.zeros([1, ], dtype=np.int)
	f3 = len(curr_w)

	# Feature 4, oh of prev word
	f4 = one_hot_with_oov(prev_w, wordToIndex)

	# Feature 5 oh of current
	f5 = one_hot_with_oov(curr_w, wordToIndex)

	# Feature 6, oh of next word
	f6 = one_hot_with_oov(next_w, wordToIndex)

	# Feature 7: os the token first in this sentence
	f7 = np.zeros([1, ], dtype=np.int)
	if targetI == 0:
		f7 = 1

	# Feature 8, 9, 10 : Gazeteer (prev, curr, next)
	f8 = np.zeros([1, ], dtype=np.int)  # prevous word
	f9 = np.zeros([1, ], dtype=np.int)  # curr word
	f10 = np.zeros([1, ], dtype=np.int)  # next word
	if prev_w is not None:
		if prev_w.lower() in gazeteer.keys():
			f8 = 1
	if curr_w is not None:
		if curr_w.lower() in gazeteer.keys():
			f9 = 1
	if next_w is not None:
		if next_w.lower() in gazeteer.keys():
			f10 = 1

	featureVector = np.hstack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
	return featureVector


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