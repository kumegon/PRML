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

def gen_Sw(m1,m2,cls_1,cls_2):
  Sw = np.zeros((2,2))
  m1 = np.matrix(m1).reshape(2,1)
  for n in range(N):
    xn = np.matrix(cls_1[n]).reshape(2,1)
    Sw += np.dot((xn-m1),(xn-m1).transpose())
  for n in range(N):
    xn = np.matrix(cls_2[n]).reshape(2,1)
    Sw += np.dot((xn-m2),(xn-m2).transpose())
  return Sw

def estimate_w(m1,m2,Sw):
  return np.dot(np.linalg.inv(Sw),np.matrix(m2-m1).reshape(2,1))

def f(x, a, b):
  return a * x + b

def f2(x,m,w):
  m = np.matrix(m).reshape(2,1)
  y = -(w[0][0]*x-np.dot(w.transpose(),m))/w[1][0]
  return y.item((0,0))

mean_1 = [1,1]
mean_2 = [30,-20]
cov_1 = [[100,49], [49, 64]]
cov_2 = [[64,49],[49,400]]

cls_1 = []
cls_2 = []

cls_1.extend(np.random.multivariate_normal(mean_1,cov_1,N))
cls_2.extend(np.random.multivariate_normal(mean_2,cov_2,N))


# Least square
X = gen_X(cls_1,cls_2)
T = gen_T()
W = estimate(X,T)

x = np.linspace(-100,100,1000)
y1 = [estimate_y(W,x) for x in x]


# Fisher's LDA
m1 = np.mean(cls_1, axis=0)
m2 = np.mean(cls_2, axis=0)
Sw = gen_Sw(m1,m2,cls_1,cls_2)
w = estimate_w(m1,m2,Sw)

a = -(w[0,0]/w[1,0])
m = (m1 + m2)/2
b = m[1] - a * m[0]

y2 = [f(x, a, b) for x in x]

y3 = [f2(x,m,w) for x in x]


pyplot.plot(x,y1,label="Least Square")
pyplot.plot(x,y2,label="Fisher's LDA")
pyplot.plot(x,y3,label="Fisher's LDA2")

pyplot.scatter(*zip(*cls_1),label="Class1",marker='o')

pyplot.scatter(*zip(*cls_2),label="Class2",marker='x')
pyplot.legend()
pyplot.show()



