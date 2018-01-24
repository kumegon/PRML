import numpy as np
from matplotlib import pyplot
import math

N = 100

def gen_X(ls_1,ls_2):
  X = np.zeros((2*N,3))
  for i in range(N):
    X[i][1] = ls_1[i][0]
    X[i][2] = ls_1[i][1]
    X[i+N][1] = ls_2[i][0]
    X[i+N][2] = ls_2[i][1]
    X[i][0] = 1
    X[i+N][0] = 1
  return X

def gen_T():
  T = np.zeros((2*N,1))
  for i in range(N):
    T[i][0] = 1
    T[i+N] = 0
  return T

def estimate(X,T):
  W = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),T)
  return W

def estimate_y(W,x):
  return -(W[0][0] + W[1][0] * x-0.5)/W[2][0]

mean_1 = [1,1]
cov = [[16,4], [4, 9]]

ls_1 = np.random.multivariate_normal(mean_1,cov,N)

mean_2 = [5,-10]

ls_2 = np.random.multivariate_normal(mean_2,cov,N)

X = gen_X(ls_1,ls_2)
T = gen_T()
W = estimate(X,T)

x = np.linspace(-10,10,1000)
y = list(map(lambda x : estimate_y(W,x),x))

pyplot.plot(x,y,label="Fisher's LDA")


pyplot.scatter(*zip(*ls_1),label="Class1",marker='o')

pyplot.scatter(*zip(*ls_2),label="Class2",marker='x')
pyplot.legend()
pyplot.show()


