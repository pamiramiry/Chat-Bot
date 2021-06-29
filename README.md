# Chat-Bot
AI chat bot that answers simple questions about a business

## Description

The AI chat bot I created is setup to answer questions about a pizza store. For the training data I used a JSON file where I wrote a bunch of statements that a user could ask and 
mapped those statements to responses the chatbot would use. The mapping was implemnted using a dictionary. With this data we would train the neural network to take a response from
the user and map it out to one the questions and give the appropriate response. In order to train the model with it we would have to make a list of all the words from all the questions then, make another list for each indivdual question which has the same length as the list of all the words and then for each word in the question we would put a 1 where the question had a word that belonged to the total list and a zero if it did not. 

I will display an image for better explanation

![image](https://user-images.githubusercontent.com/48389891/123858082-179fd200-d8f1-11eb-9ba5-655cfbed7ee7.png)

Next I just create and train the model using TensorFlow and TFlearn.

## How To Run

Make sure you download all the libraries from the requirments folder. Any version of Python should work but I used 3.9.5 and for pip I used 21.1.1


