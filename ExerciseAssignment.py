import csv

import numpy as np

file = open(r"C:\Users\mygam\Documents\DataForAssignment.csv")
csvreader = csv.reader(file)

header = []
header = next(csvreader)
rows = []
for row in csvreader:
    rows.append(row)

file.close()

rows = np.array(rows)
rows = rows.astype(np.float32)


def euclidean(vec1, vec2):
    r = np.sqrt(sum((vec1-vec2)**2))
    return r


def manhattan(vec1, vec2):
    r = sum(abs(vec1-vec2))
    return r


def cosineSimilarity(vec1, vec2):
    numerator = np.dot(vec1, vec2)
    denom = np.sqrt(sum(np.array(vec1)**2))*np.sqrt(sum(np.array(vec2)**2))
    return numerator/denom


def jaccard(vec1, vec2):

    minimus = []
    for i in range(0, len(vec1)):
        minimus.append(min(vec1[i], vec2[i]))

    maximus = []
    for i in range(0,len(vec1)):
        maximus.append(max(vec1[i], vec2[i]))

    j = sum(minimus)/sum(maximus)

    return j


def tanimoto(vec1, vec2):
    numerator= np.dot(vec1, vec2)
    denom = (sum(np.array(vec1)**2))+ \
            (sum(np.array(vec2)**2))-numerator
    return numerator/denom


firstRow = 0
secondRow = 1

def sortkey(item):
    return item[1]


def knearest(vec, data, k):
    result = []
    for row in range(0, len(data)):
        dist = euclidean(vec, data[row])
        result.append([row, dist])

    sortedResult = sorted(result, key=sortkey)
    indices = []
    if k < len(data):
        for r in range(0, k):
            indices.append(sortedResult[r][0])
    else:
        indices = sortedResult[:, 0]
    return indices


def knearestM(vec, data, k):
    result = []
    for row in range(0, len(data)):
        dist = manhattan(vec, data[row])
        result.append([row, dist])

    sortedResult = sorted(result, key=sortkey)
    indices = []
    if k < len(data):
        for r in range(0, k):
            indices.append(sortedResult[r][0])
    else:
        indices = sortedResult[:, 0]
    return indices


givenVec = rows[63]
givenVec1 = rows[63]

knn = knearest(givenVec, rows, 5)

knnM = knearestM(givenVec, rows, 5)

print("Knn euclidean", knn, "\n")

print("Knn Manhattan", knnM, "\n")

print("Euclidean distance between rows using euclidean formula: ")
print(euclidean(rows[5], rows[100]), '\n')

#print("Manhattan distance between rows using euclidean formula: ")
#print(manhattan(rows[0], rows[1]), "\n")

print('Cosine Similarity: ')
print(cosineSimilarity(rows[55], rows[95]), '\n')

print('Jaccard Similarity: ')
print(jaccard(rows[10], rows[140]), '\n')

print('Tanimoto Similarity: ')
print(tanimoto(rows[3], rows[102]), '\n')