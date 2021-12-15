def furthestCentroids(centroids):
    x = -1
    y=-1
    maxDist = -1
    for i in range(len(centroids)):
        for j in range(i+1,len(centroids)):
            dist = np.linalg.norm(centroids[i]-centroids[j])
            if dist>maxDist:
                maxDist = dist
                x = i
                y = j
    return (x,y)
                


def getTwoFurthestClusters(clusters,vectors):
    x = clusters.keys()
    centroids = []
    centroidmap = []
    for i in x:
        sum_vector = np.zeros(len(vectors[0]))
        s = 0
        for j in clusters[i]:
            sum_vector = np.add(sum_vector,vectors[j])
        for j in range(len(vectors[0])):
            sum_vector[j] = sum_vector[j]/len(vectors[0])
        centroids.append(sum_vector)
        centroidmap.append(i)
    (p,q) = furthestCentroids(centroids)
    furthestClusters = []
    furthestClusters.append(clusters[centroidmap[p]])
    furthestClusters.append(clusters[centroidmap[q]])
    return furthestClusters

def remove_empty_from_dict(d):
    if type(d) is dict:
        return dict((k, remove_empty_from_dict(v)) for k, v in d.items() if v and remove_empty_from_dict(v))
    elif type(d) is list:
        return [remove_empty_from_dict(v) for v in d if v and remove_empty_from_dict(v)]
    else:
        return d
    