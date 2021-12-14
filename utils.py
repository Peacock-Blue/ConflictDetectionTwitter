import re
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
def kmeans(k, max_iter, vectors):
    clusters = {}
    centroids = {}
    idx = np.random.choice(len(vectors), k, replace=False)
    for i in range(k):
        clusters[i] = []
        centroids[i] = vectors[idx[i]] 
    clusters = assignToCluster(clusters, vectors, centroids)
    for _ in range(max_iter-1):
        for i in range(k):
            for j in clusters[i]:
                centroids[i] = centroids[i] + vectors[j]
            if clusters[i] != []:
                centroids[i] = centroids[i] / len(clusters[i])
            if len(clusters[i]):
                clusters[i].clear()
        clusters = assignToCluster(clusters, vectors, centroids)
    return clusters


def len_counts(clusters):
    lens = [len(cluster) for cluster in clusters.values()]
    return dict(Counter(lens))


def display_unique_tweets(tweets, cluster):
    c_tweets = [tweets[i] for i in cluster]
    for i in range(len(c_tweets)):
        if not c_tweets[i] in c_tweets[:i]:
            print(c_tweets[i])


class TwitterClient(object):
    def __init__(self):
        try:
            self.auth = OAuthHandler(os.getenv('api_key'), os.getenv('api_secret'))
            self.auth.set_access_token(os.getenv('oauth_token'), os.getenv('oauth_token_secret'))
            self.api = tweepy.API(self.auth)
            assert self.api
        except:
            print("Error: Authentication Failed")
    
    
    def get_tweets(self, query, count = 10):
        tweets = []
        try:
            fetched_tweets = self.api.search_tweets(q = query, count = count)
            for tweet in fetched_tweets:
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


