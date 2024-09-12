import tweepy
import os
from dotenv import load_dotenv

class TwitterClient(object):
    def __init__(self):
        load_dotenv('auth.txt')
        try:
            self.auth = tweepy.OAuthHandler(os.getenv('api_key'), os.getenv('api_secret'))
            self.auth.set_access_token(os.getenv('oauth_token'), os.getenv('oauth_token_secret'))
            self.api = tweepy.API(self.auth)
        except Tweepy.TweepError:
            print("Error: Authentication Failed")

    def get_tweets(self, query, count = 10):
        tweets = []
        try:
            fetched_tweets = self.api.search_tweets(q = query, count = count)
            for tweet in fetched_tweets:
                if tweet.text not in tweets:
                    tweets.append(tweet.text)
            return tweets
        except tweepy.TweepError as e:
            print("Error : " + str(e))

    def get_trending_tags(self):
        trends1 = self.api.get_place_trends(1)
        data = trends1[0]
        trends = data['trends']
        names = []
        for trend in trends:
            if trend['name'][0]=='#':
                names.append(trend['name'])
        return names

    def createDataset(self, fn, tag, num):
        fn = "./Datasets/" + fn
        with open(fn, 'w') as f:
            tweets = self.get_tweets(tag, num)
            for tweet in tweets:
                f.write(tweet + "_$_")
