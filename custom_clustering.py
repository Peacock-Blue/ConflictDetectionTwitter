from utils import *

def len_counts(clusters):
    lens = [len(cluster) for cluster in clusters.values()]
    return dict(Counter(lens))

def get_avg_dist(clusterA, clusterB, vectors):
    dist_sum = 0
    for i in clusterA:
        for j in clusterB:
            dist_sum += np.sum(np.power(vectors[i] - vectors[j], 2))
    
    dist_avg = dist_sum / (len(clusterA) * len(clusterB))
    return dist_avg

def mergeOnce(clusters, vectors):
    min_dist = 2**30
    min_loc = (0, 0)
    for clusterInxA in clusters:
        for clusterInxB in clusters:
            if clusterInxA != clusterInxB:
                dist = get_avg_dist(clusters[clusterInxA], clusters[clusterInxB], vectors)
                if dist < min_dist:
                    min_loc = (clusterInxA, clusterInxB)
                    min_dist = dist

    clusters[min_loc[0]] = clusters[min_loc[0]] + clusters[min_loc[1]]
    clusters.pop(min_loc[1])

def mergeToK(clusters, vectors, K):
    while len(clusters.keys()) > K:
        mergeOnce(clusters, vectors)

def display_unique_tweets(tweets, cluster):
    c_tweets = [tweets[i] for i in cluster]
    for i in range(len(c_tweets)):
        if not c_tweets[i] in c_tweets[:i]:
            print(c_tweets[i])

def custom_clustering(tweets,k):

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
    n = len(processed_tweets)
    m = len(alphabet)
    clusters = dict()
    vectors = dict()
    for i in range(n):
        vectors[i] = vectorize(processed_tweets[i], alphabet)
        clusters[i] = [i]
    mergeToK(clusters, vectors, k)
    return (clusters,vectors)

        