import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

# try:
#     with open("data.pickle", "rb") as f:
#         words, labels, training, output = pickle.load(f)
# except:

# list of all the words
words = []
# list of the tags
labels = []
# list of all words for each specific tag
docs_x = []
# list of all tags
docs_y = []
# go through the json file and collect all the words and tags
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# stem, lower all the characters and remove duplicates from words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# create the training data
for x, doc in enumerate(docs_x):
    bag = []
    # wrds is a list of words in a intent
    # doc is a list of words inside docs_x
    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    # add the bag to the training data
    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)


# with open("data.pickle", "wb") as f:
#     pickle.dump((words, labels, training, output), f)

# tensorflow.reset_default_graph()

# create and train the model

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
#     model.load("model.tflearn")
# except:
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


# Function takes a user input and changes it to 1s and 0s
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    # tokenize and lower input
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    # check if words match with s_words if so add 1 to the bag
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


# user gives an input and chat bot gives appropriate response
def chat_bot():
    print("I am a bot")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        # get a result from the trained model
        results = model.predict([bag_of_words(inp, words)])
        # get the most likely index for the labels
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        # collect all the responses from the tag
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


chat_bot()
