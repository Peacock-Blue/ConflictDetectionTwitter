import numpy as np

class KMeans:
    def __init__(self, num, iter, vectors, alphabet):
        self.k = num
        self.max_iter = iter
        self.vectors = vectors
        self.alphabet = alphabet
        self.clusters = dict()
        self.clusters[0] = [i for i in range(len(self.vectors))]

    def closestCluster(self, vector, centroids):
        closest = -1
        minDist = 2**30
        for key in centroids:
            dist = np.linalg.norm(centroids[key] - vector)
            if dist < minDist:
                minDist = dist
                closest = key
        return closest

    def assignToCluster(self, centroids):
        for i in range(len(self.vectors)):
            c = self.closestCluster(self.vectors[i], centroids)
            self.clusters[c].append(i)
        return self.clusters

    def run(self):
        centroids = {}
        idx = np.random.choice(len(self.vectors), self.k, replace=False)
        for i in range(self.k):
            self.clusters[i] = []
            centroids[i] = self.vectors[idx[i]]
        self.clusters = self.assignToCluster(centroids)
        for _ in range(self.max_iter-1):
            for i in range(self.k):
                centroids[i] = np.zeros(len(self.alphabet))
                for j in self.clusters[i]:
                    centroids[i] = centroids[i] + self.vectors[j]
                if self.clusters[i] != []:
                    centroids[i] = centroids[i] / len(self.clusters[i])
                if len(self.clusters[i]):
                    self.clusters[i].clear()
            self.clusters = self.assignToCluster(centroids)
        return self.clusters
