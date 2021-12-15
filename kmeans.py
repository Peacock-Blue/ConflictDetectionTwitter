from utils import *


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


def kmeans_cl(tweets, k, max_iter):
    processed_tweets = [process_tweet(tweet) for tweet in tweets]
    while [] in processed_tweets:
        processed_tweets.remove([])

    freqs = dict()
    wordToTweet = dict()
    for tweet in processed_tweets:
        count_words(tweet, freqs, wordToTweet)

    freq_sorted = list(freqs.items())
    freq_sorted.sort(key = lambda x : -x[1])

    alphabet = [i[0] for i in freq_sorted]
    clusters = dict()
    vectors = dict()
    for i in range(len(processed_tweets)):
        vectors[i] = vectorize(processed_tweets[i], alphabet)
    clusters[0] = [i for i in range(len(processed_tweets))]
    clusters = kmeans(k, max_iter, vectors)
    return (clusters,vectors)