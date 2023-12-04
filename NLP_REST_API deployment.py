# -*- coding: utf-8 -*-
## Created RestAPI for various model (KNN model, PyTorch NLP model, and Tensorflow NLP model)
## Data preparation (1).

from flask import Flask, request
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn as nn
from torch.nn import functional as F

nltk.download('all')

dataset = pd.read_csv('Customer_Reviews.tsv.txt', delimiter= '\t', quoting = 4)
dataset.head()

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
dataset.info()

corpus = []
import re
for i in range(0, 3000):
  customer_review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
  customer_review = customer_review.lower()
  customer_review = customer_review.split()
  clean_review = [ps.stem(word) for word in customer_review if not word in set(stopwords.words('english'))]
  clean_review = ' '.join(clean_review)
  corpus.append(clean_review)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 1500, min_df = 3, max_df = 0.6)
X = vectorizer.fit_transform(corpus).toarray()
X
y = dataset.iloc[:, 1].values
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

with open('tfidfmodel.pickle','wb') as file:
    pickle.dump(vectorizer,file)                   ## vector saving
files.download('tfidfmodel.pickle')

**************************************************************
# KNN classifier (2)
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)

y_pred_knn = classifierKNN.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
cmknn = confusion_matrix(y_test, y_pred_knn)

# New Samples
sample = ["Today is good day, let's have a camp"]
sample = vectorizer.transform(sample).toarray()
sample
sentiment = classifierKNN.predict(sample)
sentiment

with open('textclassifier.pickle','wb') as file:
    pickle.dump(classifierKNN,file)                 ## model saving


****************************************************************
## With torch (Tensor) (3)

Xtrain_ = torch.from_numpy(X_train).float()
Xtest_ = torch.from_numpy(X_test).float()

ytrain_ = torch.from_numpy(y_train)
ytest_ = torch.from_numpy(y_test)

Xtrain_.shape, ytrain_.shape
Xtest_.shape, ytest_.shape

input_size=500
output_size=2
hidden_size=512

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = torch.nn.Linear(input_size, hidden_size)
       self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
       self.fc3 = torch.nn.Linear(hidden_size, output_size)

   def forward(self, X):
       X = torch.relu((self.fc1(X)))
       X = torch.relu((self.fc2(X)))
       X = self.fc3(X)
       return F.log_softmax(X,dim=1)
model = Net()

import torch.optim as optim
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.NLLLoss()

epochs = 200
for epoch in range(epochs):
  optimizer.zero_grad()
  Ypred = model(Xtrain_)
  loss = loss_fn(Ypred,  ytrain_)
  loss.backward()
  optimizer.step()
  print('Epoch',epoch, 'loss',loss.item())

sample = ["Today is good day, let's have a camp"]
sample = vectorizer.transform(sample).toarray()
sample

torch.from_numpy(sample).float()

sentiment = model(torch.from_numpy(sample).float())
sentiment


model.state_dict()
torch.save(model.state_dict(),'text_classifier_pytorch')       ## Pytorch NLP model saving
model.load_state_dict(torch.load('text_classifier_pytorch'))

*****************************************************************
# Pytorch RESTAPI (4)

!pip install flask-ngrok
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

with open('tfidfmodel.pickle','rb') as file:
    tfidf = pickle.load(file)

## Declare an endpoint and creat a method to read request, then predict the output

@app.route('/predict',methods=['POST'])
def customer_behavior():
    request_data = request.get_json(force=True)
    text = request_data['sentence']
    print("printing the sentence")
    print(text)
    text_list=[]
    text_list.append(text)
    print(text_list)
    numeric_text = tfidf.transform(text_list).toarray()
    output = model(torch.from_numpy(numeric_text).float())
    print("Printing predictions")
    print(output[:,0][0])
    print(output[:,1][0])
    sentiment="unknown"
    if torch.gt(output[:,0][0],output[:,1][0]):
      print("negative prediction")
      sentiment="negative"
    else:
      print("positive")
      sentiment="positive prediction"
    print("Printing prediction")
    print(sentiment)
    return "The prediction is {}".format(sentiment)
##  return sentiment
app.run()

## Click the restAPI from the Postman tool, predict the sentiment, send request.

***********************************************************************
***********************************************************************
## Integrate twitter sentiment analysis app with PyTorch RestAPI (5)

url='http://XXXXXXXX.ngrok.io/predict'   ## From Postman

import json
import requests

request_data = json.dumps({'sentence':'Good sample with nice insights'})
response = requests.post(url,request_data)        ## implement restAPI from other notebook.
response.text


import tweepy
consumer_key='XXX'
consumer_secret='XXX'
access_token ='XXX'
access_secret='XXX'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth,timeout=20)

tweet_text = 'Sunshine';
tweets_list = []
for status in tweepy.Cursor(api.search,q=tweet_text,lang='en',result_type='recent').items(300):
        tweets_list.append(status.text)

len(tweets_list)
tweets_list[1]

## Clean after extract from Tweet
import re
for i in range(len(tweets_list)):
  tweet = tweets_list[i]
  tweet = tweet.lower()
  tweet = re.sub(r'\W',' ',tweet)
  tweet = re.sub(r'\s+',' ',tweet)
  tweet = re.sub('[^a-zA-Z]',' ',tweet)
  tweets_list[i] = tweet
tweets_list[2]

positive_tweet = 0
negative_tweet = 0
for tweet in tweets_list:
  request_data = json.dumps({'sentence':tweet})  ## Invoke the restAPI from each tweet (Json)
  response = requests.post(url, request_data)    ## Get response using request.post
  sentiment = response.text
  if sentiment == 'positive':
    positive_tweet += 1
  else:
    negative_tweet += 1

positive_tweet
negative_tweet

## Modify RestAPI (4), function of customer_behavior() with "return sentiment"
## Will get different url, and update url, so analyze real time tweets.


*******************************************************************************
## Tensorflow NLP RestAPI

import tensorflow as tf
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(500, activation='relu'),
  tf.keras.layers.Dense(500, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200)
loss, accuracy =model.evaluate(X_test, y_test)
model.summary()

sample = ["Sunny is good day for children's camp"]
sample = vectorizer.transform(sample).toarray()
sample

sentiment = model.predict(sample)[:,1]
sentiment

model.save('text_classifier_model_Tensorflow')           ## model saving
!zip -r text_classifier_model.zip text_classifier_model
files.download('text_classifier_model.zip')

from tensorflow.keras.models import load_model
model = load_model('text_classifier_model_Tensorflow')

with open('tfidfmodel.pickle','rb') as file:
    tfidf = pickle.load(file)

!pip install flask-ngrok
from flask_ngrok import run_with_ngrok
from flask import Flask, request

app = Flask(__name__)
run_with_ngrok(app)
@app.route('/predict',methods=['POST'])               ## Similar as RestAPI Pytorch RESTAPI (4)
def text_classifier():                                ## Create an end point and read the incoming request.
    request_data = request.get_json(force=True)
    text = request_data['sentence']
    print("printing the sentence")
    print(text)
    text_list=[]
    text_list.append(text)
    print(text_list)
    numeric_text = tfidf.transform(text_list).toarray()
    output = model.predict(numeric_text)[:,1]
    print("Printing prediction")
    print(output)
    sentiment="unknown"
    if output[0] > 0.5 :
      print("positive prediction")
      sentiment="postive"
    else:
      print("negative prediction")
      sentiment="negative"
    print("Printing sentiment")
    print(sentiment)
    return "The sentiment is {}".format(sentiment)
app.run()

## Then get a public url using which we can predict the sentiment. Copy to postman tool and send a post request for this sentence.
## So we can also create RESTAPI using flask for the tensorflow models.

******************************************************************************
## Tensorflow model serverless
## Upload (saved weights from variables directory and other tfidf model) to bucket
## Use gcp cloud functions, set trigger type (HTTP)
## Reconstruct the rensorflow model, invoke the cloud function from inside or outside using the http triggering endpoint.
## Many thanks to udemy courses for some details.
