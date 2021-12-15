from agglo import *

def agglo_test(fname, K = 10, ThresholdFactor = 1):
    f = open(fname, 'r')
    tweets = f.read().split('_$_')
    f.close()

    clusters, vectors, processed_tweets = agglo(tweets, K)
    cluster_keys = clusters.keys()
    trimmed_clusters = dict()
    for key in clusters:
        if len(clusters[key]) != 0:
            trimmed_clusters[key] = clusters[key]

    return is_controversial(clusters, vectors, ThresholdFactor)