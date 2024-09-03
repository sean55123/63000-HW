import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)]) # Here, the matrix is built in augmented form

    for i in range(n):
        if Ab[i, i] == 0: # Do the row switching. So that the pivot won't be zero. To prevent the zero-dividing issue.
            for k in range(i + 1, n):
                if Ab[k, i] != 0:
                    Ab[[i, k]] = Ab[[k, i]]
                    break
                else:
                    raise ValueError("Matrix is singular!")

        Ab[i] = Ab[i] / Ab[i, i] # Making sure the pivot of each row is 1

        for j in range(i + 1, n): # Do the row operation to make sure the matrix is in echelon form
            Ab[j] = Ab[j] - Ab[j, i] * Ab[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1): # Do the back substitution
        x[i] = Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])

    return Ab, x

A = np.array([
    [-23, 26, -42, -32, -90],
    [-2, 1, 0, -3, -4],
    [-17, 19, -28, -22, -63],
    [-13, 14, -24, -16, -52],
    [18, -20, -32, 23, 69]], dtype="float")

b = np.array([-6, -2, -3, -2, 3])

Ab, sol = gaussian_elimination(A, b)

print("Echelon matrix:")
print(Ab)
print("Solution:", sol)