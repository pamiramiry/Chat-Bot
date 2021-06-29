# Chat-Bot
AI chat bot that answers simple questions about a business

## Description

The AI chat bot I created is setup to answer questions about a pizza store. For the training data I used a JSON file where we wrote a bunch of statements a user could ask and 
mapped those statements to responses the chatbot would use. The mapping was implemnted using a dictionary. With this data we would train the neural network to take a response from
the user and map it out to one the questions and give the appropriate response. In order to train the model with it we would have to make a list of all the words of the questions
into a list then make a list for each indivdual question which had the same length as list of all the words and then for each word in the question we would put a 1 where the 
question had a word that belonged to the total list and a zero if it didnt. 
I will displ
![image]
Next I just create and train the model using TensorFlow and TFlearn.


## How To Run

Make sure you have all the libraries from the requirments folder are downloaded. Any version of Python should work but I used 3.9.5 and for pip I used 21.1.1


