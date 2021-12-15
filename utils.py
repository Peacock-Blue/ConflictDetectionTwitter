import tweepy
import numpy as np
from tweepy import OAuthHandler
import os

import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk import download
from collections import Counter
from myconfig import *
from textblob import TextBlob

# download('stopwords')

def process_tweet(tweet):
    '''
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    tweets_eng = []
    for word in tweets_clean:
        flag = True
        for i in word:
            if ord(i) >= 256:
                flag = False
                break
        if flag:
            tweets_eng.append(word)
    return tweets_eng


def count_words(tweet:list, freqs:dict, wordToTweet:dict):
    for word in tweet:
        if word in freqs:
            freqs[word] += 1
            wordToTweet[word].append(tweet)
        else:
            freqs[word] = 1
            wordToTweet[word] = [tweet]
    return freqs

def normalize(v):
    norm = np.power(np.sum(np.power(v,2)), 0.5)
    if norm == 0:
        return 0
    return v / norm



def vectorize(tweet, alphabet):
    v = np.zeros(len(alphabet))
    for i in range(len(alphabet)):
        if alphabet[i] in tweet:
            v[i] += 1
    return v

def closestCluster(vector, centroids):
    closest = -1
    minDist = 2**30
    for key in centroids:
        dist = np.linalg.norm(centroids[key] - vector)
        if dist < minDist:
            minDist = dist
            closest = key
    return closest


def assignToCluster(clusters, vectors, centroids):
    for i in range(len(vectors)):
        c = closestCluster(vectors[i], centroids)
        clusters[c].append(i)
    return clusters


def len_counts(clusters):
    lens = [len(cluster) for cluster in clusters.values()]
    return dict(Counter(lens))


def display_unique_tweets(tweets, cluster):
    c_tweets = [tweets[i] for i in cluster]
    for i in range(len(c_tweets)):
        if not c_tweets[i] in c_tweets[:i]:
            print(c_tweets[i])


#Twitter class for fetching tweets

class TwitterClient(object):
    def __init__(self):
        try:
            self.auth = tweepy.OAuthHandler(twitterApiKey,twitterApiKeySecret)
            self.auth.set_access_token(twitterAccessToken,twitterAccessTokenSecret)
            self.api = tweepy.API(self.auth)
            assert self.api
        except:
            print("Error: Authentication Failed")
    
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
    def get_tweet_sentiment(self, tweet):
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
    
    def get_tweets(self, query, count = 10):
        tweets = []
        try:
            fetched_tweets = self.api.search_tweets(q = query, count = count)
            for tweet in fetched_tweets:
                '''
                all_english = True
                for c in tweet.text:
                    if ord(c) >= 256:
                        all_english = False
                        break
                if not all_english:
                    continue
                '''
                parsed_tweet = {}
                parsed_tweet['text'] = tweet.text
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
            return tweets
        except tweepy.TweepyException as e:
            print("Error : " + str(e))

    def fetch_tweets(self, query, count = 10):
        try:
            return self.api.search_tweets(q = query, count = count)
        except tweepy.TweepyException as e:
            print("Error : " + str(e))
            