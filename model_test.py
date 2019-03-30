import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy
import collections
import string
#
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from profanity_check import predict, predict_prob

df = pd.read_csv("train_cleaned.csv");
target = df['toxic']
df = df.drop(columns=["toxic"])
x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=0)
#LogisticRegression
model = LogisticRegression(class_weight='balanced')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print("LogisticRegression:")
print("score:"+str(score))
print("f_score:"+str(f1_score(y_test,y_pred)))
#MLP MLPClassifier
model = MLPClassifier(solver='lbfgs', alpha=1e-2,
                    hidden_layer_sizes=(5, 2), random_state=1)

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print("MLPClassifier:")
print("score:"+str(score))
print("f_score:"+str(f1_score(y_test,y_pred)))
