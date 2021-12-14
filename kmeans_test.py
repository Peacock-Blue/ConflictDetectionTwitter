from kmeans import *

def is_controversial(clusters, vectors, ThresholdFactor):
    try:
        cluster_lists = clusters.values()
        cluster_lens = [len(cluster) for cluster in cluster_lists]
        cluster_lens.sort()

        if sum(cluster_lens[:-1]) * ThresholdFactor >= cluster_lens[-1]:
            return True
        else:
            return False
    except Exception:
        print('exception occurred')
        return False

def kmeans_test(fname, K = 10, ThresholdFactor = 3, max_iter = 100):
    f = open(fname, 'r')
    tweets = f.read().split('_$_')
    f.close()

    clusters, vectors, processed_tweets = kmeans_cl(tweets, K, max_iter)
    cluster_keys = clusters.keys()
    trimmed_clusters = dict()
    for key in clusters:
        if len(clusters[key]) != 0:
            trimmed_clusters[key] = clusters[key]

    return is_controversial(clusters, vectors, ThresholdFactor)