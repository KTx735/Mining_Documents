import numpy as np

data = np.array([
                [10,3,3,5],
                [5,4,5,3],
                 [9,4,3,4],
                 [20,15,1,20]])

print(data)

def sorkey(item):
    return [1]

def knearest (vec, newData, k):

    result = []
    for row in range(0, len(data)):
        dist = euclidian(vec, data[row])
        result.append([row, dist])

    # Result contains two columns
    # The first column has 0, 1, 2, ....., len(data) length ---> len()
    # The second col contains the corresponding distances with vec

    sortedResult = sorted(result, key=sorkey)
    indices = []
    if k < len(data):
        for r in range(0, k):
            indices.append(sortedResult[r][0])
        else:
            indices = sortedResult[:, 0]

        return indices

    # 7 data points with each has 5 features
    newData = np.array([10,3,5,10],
                       [5,4,5,3,6],
                       [10, 4, 6, 4, 9],
                       [8,6,2,6,3],
                       [10, 3, 3, 5, 8],
                       [9,2,1,2,11],
                       [9,3,1,2,11],
                       [5.1,3.99,5.01,3.2,6.1])

    givenVec = data[1]

    knn = knearest(givenVec, data, 6)

    print(knn)

def euclidian(vec1, vec2):
    r = np.sqrt(sum((vec1 - vec2)**2))
    return r

def manhattan(vec1, vec2):
    r = sum(abs(vec1-vec2))
    return r

def jaccard(vec1, vec2):

    minimus = []
    for i in range(0, len(vec1)):
        minimus.append(min(vec1[i], vec2[i]))

    maximus = []
    for i in range(0,len(vec1)):
        maximus.append(max(vec1[i], vec2[i]))

    j = sum(minimus)/sum(maximus)


def cosine(vec1, vec2):
    numerator = np.dot(vec1, vec2)
    denom = np.sqrt(sum(np.array(vec1)**2))*np.sqrt(sum(np.array(vec2)**2))
    return numerator/denom

def tanimoto(vec1 , vec2):
        numerator = np.dot(vec1, vec2)
        denom = (sum(np.array(vec1)**2)) + (sum(np.array(vec2)**2)) - (vec1 * vec2)
        return numerator/denom

firstRow = 0
secondRow = 2

print("Distance between row using Euclidean formula", firstRow, "and row", secondRow )
print(euclidian(data[firstRow], data[secondRow]))

print("Distance between row using Manhattan formula ", firstRow, " and row ", secondRow)
print(manhattan(data[firstRow], data[secondRow]))

print("Jaccard similarity", firstRow, "and row", secondRow)
print(jaccard(data[firstRow], data[secondRow]))

print("Cosine similarity", firstRow, "and row", secondRow)
print(cosine(data[firstRow], data[secondRow]))

print('Using tanimoto. first row:', firstRow, "and second row: ", secondRow)
print(tanimoto(data[firstRow], data[secondRow]))