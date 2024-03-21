import numpy as np
import matplotlib.pyplot as plt
import pickle

with open ('8.8_eigen.pkl', 'rb') as f:
    X = pickle.load(f)

# plt.plot(X[:,0], X[:,1], 'x')
# plt.axis('equal')
# plt.show()


mean_val = X.mean(axis=0)
N = X.shape[0]

X_centered = X.copy() - mean_val

# показ отцентрированных данных
# plt.plot(X_centered[:,0], X_centered[:,1], 'x')
# plt.axis('equal')
# plt.show()

X_cov = X_centered.T.dot(X_centered) / (N - 1)

# print(f'Ковариационная матрица: \n{X_cov}')

# # готовая реализация в numpy

# print(f'Ковариационная матрица: \n{np.cov(X_centered, rowvar=0)}')

from numpy import linalg

eigenvalues, eigenvectors = linalg.eig(X_cov)

# for i in range(eigenvalues.size):
#     print(f'lambda_{i} = {eigenvalues[i]}, w = {eigenvectors[i]}')
origin = X_centered.mean(axis=0)
# # *np.tile(origin, (2, 1)) = [origin[0], origin[0]], [origin[1], origin[1]]
# plt.quiver(*np.tile(origin, (2, 1)), *eigenvectors.T,
#             color=['r','b','g'],
#             scale=eigenvalues)
# plt.plot(X_centered[:,0], X_centered[:,1], 'x')
# plt.show()
max_eigenval = np.argmax(eigenvalues)
max_eigenvec = eigenvectors[:,max_eigenval].reshape(-1,1)

X_redused = X.dot(max_eigenvec)
# plt.plot(X_redused, np.zeros(N), 'x')
# plt.show()

X_inverse = X_redused.dot(max_eigenvec.reshape(1,-1))

plt.plot(X_inverse[:,0], X_inverse[:,0], 'o')
plt.show()