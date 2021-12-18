import numpy as np

import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk import download

class PreProcessor():
    def __init__(self, data):
        self.tweets = data

    def preProcessor(self, tweet):
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
    def preProcess(self):
        processed_tweets = [self.preProcessor(tweet) for tweet in self.tweets]
        while [] in processed_tweets:
            processed_tweets.remove([])
        return processed_tweets

class CreateFreqs():
    def __init__(self, p):
        self.processed_tweets = p
        self.freqs = dict()
        self.wordToTweet = dict()

    def count_words(self, tweet):
        for word in tweet:
            if word in self.freqs:
                self.freqs[word] += 1
                self.wordToTweet[word].append(tweet)
            else:
                self.freqs[word] = 1
                self.wordToTweet[word] = [tweet]
        return self.freqs

    def createFreqs(self):
        for tweet in self.processed_tweets:
            self.count_words(tweet)
        return self.freqs

    def createFreqSorted(self):
        freq_sorted = list(self.freqs.items())
        freq_sorted.sort(key = lambda x : -x[1])
        return freq_sorted

class Vectorizer:
    def __init__(self, p, freq_sorted):
        self.vectors = dict()
        self.alphabet = [i[0] for i in freq_sorted]
        self.processed_tweets = p

    def vectorize(self, tweet):
        v = np.zeros(len(self.alphabet))
        for i in range(len(self.alphabet)):
            if self.alphabet[i] in tweet:
                v[i] += 1
        return v

    def textVectorize(self):
        for i in range(len(self.processed_tweets)):
            self.vectors[i] = self.vectorize(self.processed_tweets[i])
        return self.vectors
