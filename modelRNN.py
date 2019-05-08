import numpy as np
import json
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.layers.embeddings import Embedding
from tensorflow.python.keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from sklearn.model_selection import train_test_split

def load_data():
    # global reviewVectorData, metaVectorData
    # global data
    # global courseCode
    print("Loading all required data............")
    f = open("courseCode.json")
    courseCode = json.load(f) #Array of dicts
    f.close()
    
    f = open("data.json")
    data = json.load(f) #Array of dicts
    f.close()
    
    reviewVectorData = np.genfromtxt("reviewVectorData.csv", delimiter=",")
    metaVectorData = np.genfromtxt("metaVectorData.csv", delimiter=",")

    return reviewVectorData, metaVectorData, data, courseCode

def generate_courseMap(reviewVectorData, metaVectorData, data, courseCode):
    docs = {}
    labelDoc = {}

    for i in range(len(courseCode)):
        if courseCode[i] not in docs:
            docs[courseCode[i]] = np.concatenate((reviewVectorData[i], metaVectorData[i]))
            labelDoc[courseCode[i]] = np.array(data[i]['avgGrade'])
        else:
            temp = np.concatenate((reviewVectorData[i], metaVectorData[i])) 
            docs[courseCode[i]] = np.append(docs[courseCode[i]], temp)
            labelDoc[courseCode[i]] = np.append(labelDoc[courseCode[i]], data[i]['avgGrade'])
    labels = []
    for i in labelDoc.values():
        labels.append(np.mean(i))
    return docs, np.array(labels)
    
def calc_max_tokens(encoded_docs):
    num_tokens = [len(tokens) for tokens in encoded_docs]
    num_tokens = np.array(num_tokens)
    
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    return max_tokens, num_tokens

def calc_data_coverage(max_token, num_tokens):
    return np.sum(num_tokens < max_token) / len(num_tokens)

def linear_regression_equality(y_true, y_pred):
    accepted_diff = 0.3
    diff = K.abs(y_true-y_pred)
    return K.mean(K.cast(diff < accepted_diff, tf.float32))


def run():
    reviewVectorData, metaVectorData, data, courseCode =  load_data()
    print("Course Codes loaded " + str(len(courseCode)))
    print("Data loaded " + str(len(data)))
    print(reviewVectorData.shape)
    print(metaVectorData.shape)


    courseMap, courseAvgGrades = generate_courseMap(reviewVectorData, metaVectorData, data, courseCode)
    courseVector = list(courseMap.values())
    print("Generated course_map of size: ", len(courseMap))
    print(courseAvgGrades.shape)
    max_token, num_tokens = calc_max_tokens(courseVector)


    print("Truncating course vector will retain ", calc_data_coverage(max_token, num_tokens)*100, "% of the data")
    padded_docs = pad_sequences(courseVector, maxlen=max_token, padding='pre')
    print("Padded vecor shape ", padded_docs.shape)

    X_train, X_test, y_train, y_test = train_test_split(padded_docs, courseAvgGrades,
                                                        test_size=0.10,
                                                        random_state=42)
    print("Train test split ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    vocab_size = int(X_train.max()+1)
    print("Vocabulary size ", vocab_size)


    # define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=13, input_length=max_token))
    model.add(LSTM((13), batch_input_shape=(None, 1433, 13), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[linear_regression_equality])
    # summarize the model
    print(model.summary())

    model.fit(X_train, y_train, epochs=10, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('Accuracy: %f' % (accuracy*100))
    print('Loss: %f' % (loss*100))

    print(len(X_test))
    sIdx= 10
    eIdx = 21

    guess = model.predict(x=X_test[sIdx:eIdx])
    print(guess)

    for i in range(len(padded_docs)):
        for target in X_test[sIdx:eIdx]:
            if np.array_equal(target, padded_docs[i]):
                print(data[i]["code"], data[i]["courseName"], data[i]["avgGrade"])

if __name__ == "__main__":
    run()