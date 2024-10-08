# -*- coding: utf-8 -*-
"""spam classifier.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MJYIhjeopEdm1kQJjUMzA1ccLtGSZ8LR
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load Data Viz Pkgs
import seaborn as sns

df = pd.read_csv('mail_data.csv')
df.head()

# Value Counts
df['Category'].value_counts()

data = df.where((pd.notnull(df)),'')

data.head(10)

# Plot
sns.countplot(x='Category',data=df)

data.info()

data.shape

data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

"""Allocating the dataset categories as '0' & '1' as assigned to spam and not spam(ham) messages"""

X = data['Message']
Y = data['Category']

print(X)

print(Y)

"""0 -> spam messages

1 -> not spam messages
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

"""data set divided into 70% / 30% as train and test dataset"""

print(X.shape)
print(X_train.shape)
print(X_test.shape)

"""this are the values of the divided dataset of X features"""

print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)

"""same dataset divided values of Y features"""

features_extraction = TfidfVectorizer(min_df = 1, stop_words='english')

"""Transforming test data into feature extraction of the text

difference (minimum) = 1

stop_words = the words that can be removed which doesn't impact the meaning of the sentences(and, the)
"""

X_train_features = features_extraction.fit_transform(X_train)
X_test_features = features_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

pip install neattext

import neattext.functions as nfx

df['clean_message'] = df['Message'].apply(nfx.remove_stopwords)

df

print(X_train)

"""This is the X_train dataset divided"""

print(X_train_features)

"""this is the accuracy values of the trained dataset"""

model = LogisticRegression()

model.fit(X_train_features, Y_train)

predict_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, predict_training_data)

print('accuracy is : ', accuracy_on_training_data)

"""this is the accuracy for the training data"""

predict_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, predict_test_data)

print('accuracy is : ', accuracy_on_test_data)

"""This is the accuracy of test data"""

input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]
input_data_features = features_extraction.transform(input_mail)

prediction = model.predict(input_data_features)
print(prediction)

"""This is the final prediction line

0 -> is the spam mail


1 -> is the ham mail(not spam)
"""

#adding the names to the values [0,1] as [spam, ham]
if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')
  print(prediction)

"""Here the condition says

if part :

if the value of [0] is == (equal to) 1
then the output should be ham mail i,e not spam

else part :

if the value of [0] is  (not equal to) 1
then the output should be spam mail.
"""

