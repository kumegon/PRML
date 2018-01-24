import numpy as np
from matplotlib import pyplot
import math


P_N = 2
N = 100

a = [50, 3]
w = [1, 1]

#paramater
beta = 16.0
S = 1 / beta
alpha = 2.0

def y(x, w):
  return w[0] + w[1] * x

#a = -0.3 0.5
#beta = 0.2^2
#alpha = 2
def f(x, a):
  return a[0] + a[1] * x

def gen_date(y,cov):
  return np.random.normal(y, cov, 1)


def design_matrix(x, w):
  d_m = np.zeros((N, P_N))
  d_m[:,0] = w[0]
  for i in range(N):
    d_m[i, 1] = w[1] * x[i]
  return d_m


def sigma(alpha, beta, d_m):
  return np.linalg.inv(alpha * np.identity(P_N) + beta * np.dot(d_m.transpose(),d_m))

def average(beta, s, d_m, t):
  return beta * np.dot(np.dot(s,d_m.transpose()),t)

def estimate(d_m, t, lambda_):
  return np.dot(np.dot(np.linalg.inv(lambda_ * np.identity(P_N) + np.dot(d_m.transpose(),d_m)),d_m.transpose()),t)
  #return np.linalg.inv(lambda_ * np.identity(P_N) + np.dot(np.dot(np.dot(d_m.transpose(),d_m),d_m.transpose()),t))

x = np.linspace(-1, 1 , N)
y = list(map(lambda x : f(x, a), x))
t = list(map(lambda y : gen_date(y, S),y))

d_m = design_matrix(x, w)
s = sigma(alpha, beta, d_m)
ave = average(beta, s, d_m, t)

w_ml = estimate(d_m, t, alpha/beta)
y_ = list(map(lambda x : f(x, w_ml), x))


print(w_ml)
print(a)

pyplot.plot(x,t, label="gen")
pyplot.plot(x,y, label="real")
pyplot.plot(x,y_,label="estimate")
pyplot.legend()
pyplot.show()
