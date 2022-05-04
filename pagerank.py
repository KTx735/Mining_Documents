import numpy as np
b = 0.8
t = 0.00000000000001

#Yahoo, amazon, msoft

A=[[1, 1, 0],
   [1, 0, 0],
   [0, 1, 1]]

fig1 = [[1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0]]

fig2 = [[0, 0, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1]]

fig3 = [[1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1]]

fig4 = [[0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1]]

def randomTeleportRet(A, beta):
    threshold = 0.00000000000001
    arr= np.array(A, dtype=float)

    s=[]
    for i in range(0, len(A)):
        s.append(np.sum(arr[:,i]))

    #print(s)

    # s contains column-wise summations

    M=arr

    for j in range(0, len(A)):
        M[:,j]=M[:,j]/s[j]


    r = (1.0+np.zeros([len(M), 1]))/len(M)
    uniD = (1.0-beta)*r

    #print("\nr is :\n", np.around(r, decimals=4))
    #print("\nuniD is :\n", np.around(uniD, decimals=4))


    rprev = r
    for i in range(0, 1000):
        r= beta*np.matmul(M, rprev)+uniD

        diff = sum(abs(r-rprev))[0]
        #print(i, " ", diff)

        if diff<=threshold:
            print("\nValue of i=", i)
            break
        rprev=r
    return r

def randomTeleport(A, beta):
    threshold = 0.00000000000001
    arr= np.array(A, dtype=float)

    s=[]
    for i in range(0, len(A)):
        s.append(np.sum(arr[:,i]))

    #print(s)

    # s contains column-wise summations

    M=arr

    for j in range(0, len(A)):
        M[:,j]=M[:,j]/s[j]


    r = (1.0+np.zeros([len(M), 1]))/len(M)
    uniD = (1.0-beta)*r

    #print("\nr is :\n", np.around(r, decimals=4))
    #print("\nuniD is :\n", np.around(uniD, decimals=4))


    rprev = r
    for i in range(0, 1000):
        r= beta*np.matmul(M, rprev)+uniD

        diff = sum(abs(r-rprev))[0]
        #print(i, " ", diff)

        if diff<=threshold:
            print("\nValue of i=", i)
            break
        rprev=r

    print("\nfinal r is:\n", np.around(r, decimals=4))
    print("\n==============================")


print("Default iteration:")
randomTeleport(A, 0.8)

print("Question 1:")
randomTeleport(fig4, 0.5)

print("Question 2:")
print("Beta 1.0:")
randomTeleport(fig1, 1.0)
print("Beta 0.6:")
randomTeleport(fig1, 0.6)

print("Question 3:")
randomTeleport(fig4, 0)

print("Question 4:")
rt = np.ndarray.flatten(randomTeleportRet(fig3, 0.5))
print(rt[3] > rt[1] > rt[0] > rt[2] > rt[4])
print(rt[2] > rt[0] > rt[3] > rt[1] > rt[4])
print(rt[0] > rt[2] > rt[1] > rt[4] > rt[3])
print(rt[2] > rt[0] > rt[1] > rt[4] > rt[3])
print(rt[3] > rt[4] > rt[1] > rt[2] > rt[0])
print(rt[4] > rt[1] > rt[2] > rt[0] == rt[3])
print(rt[0] > rt[2] > rt[4] > rt[1] > rt[3])
print(rt[3] > rt[4] > rt[1] > rt[0] > rt[2])

print("Question 5:")
randomTeleport(fig2, 0)

print("Question 8:")
randomTeleport(fig1, 0.6)

print("Last Question:")
rt = np.ndarray.flatten(randomTeleportRet(fig2, 1.0))
rf = np.ndarray.flatten(randomTeleportRet(fig2, 0.6))
print(rt[0]>rf[0])
print(rt[4]>rf[4])
print(rt[1]>rf[4])
print(rt[3]>rf[3])
print(rt[4]>rf[3])