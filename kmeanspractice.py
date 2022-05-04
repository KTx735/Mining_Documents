import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


def randIndex(known, result):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(0, len(result) - 1):
        for j in range(i + 1, len(result)):
            if result[i] == result[j]:
                # Positive
                if known[i] == known[j]:
                    # True Positive
                    tp = tp + 1
                else:  # False
                    fp = fp + 1
            else:  # Negative
                if known[i] != known[j]:
                    tn = tn + 1
                else:  # False Negative
                    fn = fn + 1

    ri = (tp + tn) / (tp + fp + tn + fn)
    return ri


def purity(assignments, known):
    aLabels = set(assignments)  # Assignment Labels = aLabels
    aLabels = list(aLabels)
    kLabels = set(known)  # Known Labels = kLabels
    kLabels = list(known)

    maxOverlap = []

    for clusterIDs in aLabels:
        indicesOfClusterIDs = [ii for ii in range(0, len(assignments))
                               if assignments[ii] == clusterIDs]
        overlap = []

        for knownID in kLabels:
            indicesOfKnownIDs = [ii for ii in range(0, len(known))
                                 if known[ii] == knownID]

            overlap.append(len(set(indicesOfClusterIDs).intersection(indicesOfKnownIDs)))

        maxOverlap.append(max(overlap))

    p = sum(maxOverlap) / len(assignments)
    return p


a = [2, 1, 1, 3, 2, 2, 2, 3, 2, 1, 1, 3, 3, 2]
knw = [1, 2, 2, 2, 1, 1, 1, 3, 2, 2, 1, 3, 1, 2]

r = randIndex([0, 0, 1, 1, 1], [2, 2, 2, 3, 3])
print("Rand Index:", r, "\n")

d1 = np.random.rand(10, 2)
d2 = 2 + np.random.rand(10, 2)
d3 = 5 + np.random.rand(10, 2)
d4 = np.random.rand(10, 2)

data = np.append(d1, d2, axis=0)
data = np.append(data, d3, axis=0)
data = np.append(data, d4, axis=0)

k = 3

km = KMeans(n_clusters=k).fit(data)

print("The cluster assignments are: ")
print(km.labels_, "\n")

print("Centroids: ")
print(km.cluster_centers_, "\n")

print("Sum of squared distance: ")
print(km.inertia_)

known1 = [1] * 10
known2 = [2] * 10
known3 = [3] * 10
known4 = [1] * 10

known = known1 + known2 + known3 + known4

ri = randIndex(known, km.labels_)

# Average Silhouette Coefficient
asc = silhouette_score(data, km.labels_)

print("\nRand Index is: ", ri)

print("Assignment/Cluster data: ", a)
print("Known data: ", knw)
print("\nThe purity is: ", purity(a, knw))

print("\nASC: ", asc)

samp_sil = silhouette_samples(data, km.labels_)