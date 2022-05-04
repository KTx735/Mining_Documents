import numpy as np
from sklearn.neighbors import NearestNeighbors


def eucledian(vec1, vec2):
    r = np.sqrt(sum((vec1 - vec2) ** 2))
    return r


def sortkey(item):
    return item[1]


def knearest(vec, data, k):
    result = []
    for row in range(0, len(data)):
        dist = eucledian(vec, data[row])
        result.append([row, dist])

    # Result contains two columns.
    # The first column has 0, 1, 2, ...., len(data)-1
    # The second col contains the corresponding
    # distances with vec

    sortedResult = sorted(result, key=sortkey)
    indices = []
    if k < len(data):
        for r in range(0, k):
            indices.append(sortedResult[r][0])
    else:
        indices = sortedResult[:, 0]
    return indices


# 7 Data points with each has 5 features
data = np.array([[10, 3, 3, 5, 10],
                 [5, 4, 5, 3, 6],
                 [10, 4, 6, 4, 9],
                 [8, 6, 2, 6, 3],
                 [10, 3, 3, 5, 8],
                 [9, 2, 1, 2, 11],
                 [9, 3, 1, 2, 11],
                 [5.1, 3.99, 5.01, 3.2, 6.1]])

givenVec = data[7]

knn = knearest(givenVec, data, 5)

print(knn)

nbrs = NearestNeighbors(n_neighbors=4, algorithm='brute').fit(data)
distances, indices = nbrs.kneighbors([givenVec])

#print("using sklearn knn:")
#print(indices)
