import pandas as pd
from sklearn.linear_model import LogisticRegression
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
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from profanity_check import predict, predict_prob

df = pd.read_csv("train.csv");
df = df.drop(['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)
df['num_chars'] = df["comment_text"].apply(len)
df['words'] = df['comment_text'].apply(lambda x: len(x.split()))
df['prop_words'] = df['words']/df['num_chars']
df['capitals'] = df['comment_text'].apply(lambda x: sum (1 for char in x if char.isupper()))
df['prop_capitals'] = df['capitals']/df['num_chars']
df['prop_caps_vs_words'] = df['capitals']/df['words']
df['paragraphs'] = df['comment_text'].apply(lambda x: x.count('\n'))
df['prop_paragraphs'] = df['paragraphs']/df['num_chars']
df['prop_paragraphs_vs_words'] = df['paragraphs']/df['words']
#nltk.download('stopwords')
stopwords =  set(stopwords.words("english"))
df['num_stopwords'] = df['comment_text'].apply(lambda x: sum(x.count(w) for w in stopwords))
df['prop_stopwords']=df['num_stopwords']/df['num_chars']
df['prop_stopwords_vs_words'] = df['num_stopwords']/df['words']

df['exclamation'] = df['comment_text'].apply(lambda x: x.count("!"))
df['prop_exclamation']=df['exclamation']/df['num_chars']
df['prop_exclamation_vs_words'] = df['exclamation']/df['words']

df['question_marks'] = df['comment_text'].apply(lambda x: x.count("?"))
df['prop_question']=df['question_marks']/df['num_chars']
df['prop_question_vs_words'] = df['question_marks']/df['words']

df['punctuation'] = df['comment_text'].apply(lambda x: sum(x.count(w) for w in string.punctuation))
df['prop_punctuation']=df['punctuation']/df['num_chars']
df['prop_punctuation_vs_words'] = df['punctuation']/df['words']

df['unique_words'] = df['comment_text'].apply(lambda x: len(set(w for w in x.split())))
df['prop_unique']=df['unique_words']/df['num_chars']
df['prop_unique_vs_words'] = df['unique_words']/df['words']

repeated_threshold = 10
def num_repeated(text):
    sptext = text.split()
    word_counts = collections.Counter(sptext)
    return sum(count for word, count in sorted(word_counts.items()) if count>repeated_threshold)

df['repeated_words'] = df['comment_text'].apply(lambda x: num_repeated(x))
df['toxic_count'] = df['comment_text'].apply(lambda x: sum(predict(x.split())))
df['prop_repeated']=df['repeated_words']/df['num_chars']
df['prop_repeated_vs_words'] = df['repeated_words']/df['words']

df['mentions'] = df['comment_text'].apply(lambda x: x.count("User:"))
df['prop_mentions']=df['mentions']/df['num_chars']
df['prop_mentions_vs_words'] = df['mentions']/df['words']

sid = SentimentIntensityAnalyzer()
polarity_scores = df['comment_text'].apply(lambda x: sid.polarity_scores(x))
print(polarity_scores)
df['sentiment_compound'] = [p['compound'] for p in polarity_scores]
df['sentiment_positive'] = [p['pos'] for p in polarity_scores]
df['sentiment_negative'] = [p['neg'] for p in polarity_scores]
df['sentiment_neutral'] = [p['neu'] for p in polarity_scores]
target = df['toxic']
df = df.drop(columns=['id','comment_text'])
df.to_csv("train_cleaned.csv")
target = df['toxic']
df = df.drop(columns=["toxic"])
print(df)
# x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=0)
# model = LogisticRegression(class_weight='balanced')
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# score = model.score(x_test, y_test)
# print(score)
# print(f1_score(y_test,y_pred))
