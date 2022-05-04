import csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

data = open(r"C:\Users\mygam\Downloads\practicedata.csv")
dataread = csv.reader(data)
goldest = open(r"C:\Users\mygam\Downloads\practicegoldset.csv")
goldestread = csv.reader(goldest)

header1 = []
header1 = next(dataread)
rows1 = []
for row in dataread:
    rows1.append(row)

data.close()

header2 = []
header2 = next(goldestread)
rows2 = []
for row in goldestread:
    rows2.append(row)

goldest.close()

rows1 = np.array(rows1)
rows1 = rows1.astype(np.float32)

rows2 = np.array(rows2)
rows2 = rows2.astype(np.float32)

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

r = randIndex()
asc = silhouette_score()
p = purity()

print("Rand Index:", r, "\n")
print("The purity is: ", p)

