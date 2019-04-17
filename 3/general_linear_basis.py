import numpy as np
from matplotlib import pyplot
import math

CNT = 20
M = 10
N = 1000
RANGE = 20
cov = 0.25
alpha = 0.0004
beta = cov
regular = alpha / beta

def DesignMatrix(phys, xs):
  design_matrix = np.zeros((N,M));
  for i in range(N):
    design_matrix[i] = [phy(xs[i]) for phy in phys]
  return design_matrix

def gen_date(ys, cov):
  return np.random.normal(ys,cov)

def f(x,w,phys):
  return [phy(x) for phy in phys] @ w

def y(x):
  return 3*x + 5

def estimate_w(design_matrix, ts, regular):
  return np.linalg.solve(regular * np.identity(M) + design_matrix.T @ design_matrix, design_matrix.T @ ts)

def S(d_m):
  return np.linalg.inv(alpha * np.identity(M) + d_m.T @ d_m)

def m(S, d_m, ts):
  return beta * S @ d_m.T @ ts

def E(m, d_m, ts):
  return beta * (ts - d_m @ m).T @ (ts - d_m @ m) / 2 + alpha * m.T @ m / 2

def evidence(m, S, d_m, ts):
  return M*np.log(alpha)/2 + N*np.log(beta)/2 - E(m, d_m, ts) - np.log(np.linalg.det(np.linalg.inv(S)))/2 - N*np.log(2*np.pi)/2



xs = np.linspace(-RANGE,RANGE,N)
ys = [y(x) for x in xs]
ts = [gen_date(y, cov) for y in ys]


evis = []

for M in range(1,CNT):
  w = np.random.normal(0,10,M)
  phys = []
  for i in  range(M):
    phys.append((lambda i: lambda x:pow(x,i))(i))
  d_m = DesignMatrix(phys,xs)
  w_ml = estimate_w(d_m,ts,regular)
  sigma = S(d_m)
  mn = m(sigma, d_m, ts)
  evi = evidence(mn, sigma, d_m, ts)
  evis.append(evi)

M = np.argmax(evis) + 1
w = np.random.normal(0,10,M)
phys = []
for i in  range(M):
  phys.append((lambda i: lambda x:pow(x,i))(i))
d_m = DesignMatrix(phys,xs)
w_ml = estimate_w(d_m,ts,regular)
ys_ = [f(x,w_ml,phys) for x in xs]

print(M)
for i in range(M):
  print("+")
  print(w_ml[i])
  print("x^"+str(i))

fig, (axL, axR) = pyplot.subplots(ncols=2, figsize=(10,4))

axL.plot(range(1,CNT),evis, label="evidence")


axR.plot(xs,ys, label="real",c="blue")
axR.plot(xs,ys_,label="estimate",c="red")
pyplot.legend()
pyplot.show()
