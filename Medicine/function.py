import numpy as np
a = [[1, 0], [0, 1],[2,3]]
b = [[4, 1,3], [2, 2,2]]
print(np.dot(a, b))
print(np.multiply(a,b))
a = np.arange(3*4*5*6).reshape((3,4,5,6))
b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
print(a)
print(b)


def matrixMul(A, B):
    res = [[0] * len(B[0]) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                res[i][j] += A[i][k] * B[k][j]
    return res


def matrixMul2(A, B):
    return [[sum(a * b for a, b in zip(a, b)) for b in zip(*B)] for a in A]