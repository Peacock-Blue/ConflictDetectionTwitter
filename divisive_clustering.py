from utils import *

def findMostDistant(cluster, vectors):
    if len(cluster)<=1:
        return -1
    maxDist = -1
    item = 0
    n = len(cluster)
    for i in cluster:
        dist = 0
        for j in cluster:
            dist += np.linalg.norm(vectors[i] - vectors[j])
        dist /= n
        if dist>maxDist:
            maxDist = dist
            item = i
    return item

def isCloserToFirstCluster(cluster1, cluster2, item, vectors):
    dist1 = 0
    dist2 = 0
    n1 = len(cluster1)
    n2 = len(cluster2)
    for i in cluster1:
        dist1 += np.linalg.norm(vectors[i] - vectors[item])
    dist1 /= n1
    for i in cluster2:
        dist2 += np.linalg.norm(vectors[i] - vectors[item])
    dist2 /= n2
    if dist1<=dist2:
        return True
    return False

def dClusteringOnce(clusters, vectors):
    tmp = []
    for key in clusters:
        tmp.append(key)
    for key in tmp:
        item = findMostDistant(clusters[key], vectors)
        if item == -1:
            continue
        if item==key:
            item = clusters[key][1]
            clusters[item] = clusters[key][1:]
            for i in clusters[key][1:]:
                if isCloserToFirstCluster(clusters[item], clusters[key], i, vectors):
                    clusters[item].append(i)
                    clusters[key].remove(i)
            continue
        clusters[item] = [item]
        clusters[key].remove(item)
        for i in clusters[key]:
            if isCloserToFirstCluster(clusters[item], clusters[key], i, vectors):
                clusters[item].append(i)
                clusters[key].remove(i)

def dClusteringMinK(clusters, vectors, k):
    while len(clusters)<k:
        dClusteringOnce(clusters, vectors)

def divisive_clustering(tweets,k):
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

    dClusteringMinK(clusters, vectors, k)
    return (clusters,vectors)