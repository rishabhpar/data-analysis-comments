import xgboost as xgb
import numpy as np
import collections
import string
import pandas as pd
import pickle

import nltk
from nltk.corpus import stopwords
from nltk import pos_tag

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from profanity_check import predict, predict_prob

nltk.download('vader_lexicon')
nltk.download('stopwords')

with open('XGB_model.p','rb') as in_file:
    model = pickle.load(in_file)

def predict_comment(request):
    #Expect request_json to have single key 'comment_text'
    request_json = request.get_json(silent=True,force=True)
    
    #Calculate features
    df = generate_features(request_json)

    prediction = model.predict(xgb.DMatrix(df),ntree_limit = model.best_ntree_limit)

    return str(prediction)


def num_repeated(text):
        repeated_threshold = 10
        sptext = text.split()
        word_counts = collections.Counter(sptext)
        return sum(count for word, count in sorted(word_counts.items()) if count>repeated_threshold)

def generate_features(text_json):
    global stopwords
    df = pd.DataFrame(text_json,index=[0])
    print("text_json: ",text_json)
    df['num_chars'] = df["comment_text"].apply(len)
    df['words'] = df['comment_text'].apply(lambda x: len(x.split()))
    df['prop_words'] = df['words']/df['num_chars']
    df['capitals'] = df['comment_text'].apply(lambda x: sum (1 for char in x if char.isupper()))
    df['prop_capitals'] = df['capitals']/df['num_chars']
    df['prop_caps_vs_words'] = df['capitals']/df['words']
    df['paragraphs'] = df['comment_text'].apply(lambda x: x.count('\n'))
    df['prop_paragraphs'] = df['paragraphs']/df['num_chars']
    df['prop_paragraphs_vs_words'] = df['paragraphs']/df['words']
    
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
    
    df['repeated_words'] = df['comment_text'].apply(lambda x: num_repeated(x))
    df['toxic_count'] = df['comment_text'].apply(lambda x: sum(predict(x.split())))
    df['prop_repeated']=df['repeated_words']/df['num_chars']
    df['prop_repeated_vs_words'] = df['repeated_words']/df['words']

    df['mentions'] = df['comment_text'].apply(lambda x: x.count("User:"))
    df['prop_mentions']=df['mentions']/df['num_chars']
    df['prop_mentions_vs_words'] = df['mentions']/df['words']

    sid = SentimentIntensityAnalyzer()
    polarity_scores = df['comment_text'].apply(lambda x: sid.polarity_scores(x))
    df['sentiment_compound'] = [p['compound'] for p in polarity_scores]
    df['sentiment_positive'] = [p['pos'] for p in polarity_scores]
    df['sentiment_negative'] = [p['neg'] for p in polarity_scores]
    df['sentiment_neutral'] = [p['neu'] for p in polarity_scores]
    df = df.drop(columns=['comment_text'])

    return df



