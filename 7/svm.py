import numpy as np
from matplotlib import pyplot
import math
import time

N = 10
M = 10
CLASS = 3
RANGE = 20
ZERO = 0

def gen_X(ls_1,ls_2):
  X = np.zeros((CLASS*N,2))
  for i in range(N):
    X[i][0] = ls_1[i][0]
    X[i][1] = ls_1[i][1]
    X[i+N][0] = ls_2[i][0]
    X[i+N][1] = ls_2[i][1]
    X[i+2*N][0] = ls_2[i+N][0]
    X[i+2*N][1] = ls_2[i+N][1]
  return X

def gen_T():
  T = np.zeros((CLASS*N,1))
  for i in range(N):
    T[i][0] = 1
    T[i+N] = -1
    T[i+2*N] = -1
  return T

  return y.item((0,0))


class SVM:
  def __init__(self, X, T, kernel="gaussian", eps = 1e-5, loop_max = 1e4):
    self.X = X
    self.T = T
    self.kernel = kernel
    self.a = [0 for i in range(CLASS*N)]
    self.b = 0
    self.C = 1e0
    self.sv_index = [i for i in range(CLASS*N)]
    self.eps = eps
    self.cnt = 0
    self.tol = 1e-3
    self.loop_max = loop_max
    self.bound = 0
    self.SMO()
    self.support_vector()
    self.sv = [self.X[i] for i in self.sv_index]
    self.decision_b()

  def k(self, x, y):
    if(self.kernel == "gaussian"):
      gamma = 0.01
      return np.exp(gamma * ((x-y).transpose() @ (x-y))/ (-2))
    elif(self.kernel == "linear"):
      return x.transpose() @ y

  def y(self, x):
    sm = 0
    for i in self.sv_index:
      sm += self.a[i] * self.T[i] * self.k(x, self.X[i])
    sm += self.b
    return sm

  def is_KKT(self, i):
    a = self.a[i]
    t = self.T[i]
    x = self.X[i]
    yx = self.y(x)
    if(a == ZERO and t * yx >= 1):
      return True
    elif(a > ZERO and a < self.C and t * yx == 1):
      return True
    elif(a == self.C and t * yx <= 1):
      return True
    else:
      return False

  def check_KKT(self):
    check = True
    for i in range(CLASS*N):
      check &= self.is_KKT(i)
    return check



  def SMO(self):
    numChanged = 0
    examineAll = 1
    while(numChanged > 0 or examineAll):
      if(self.cnt > self.loop_max):
        return
      self.cnt+=1
      numChanged = 0
      if(examineAll):
        for i in range(CLASS*N):
          numChanged += self.examineExample(i)
      else:
        for i in range(CLASS*N):
          if(self.a[i] < 0 or self.a[i] > self.C):
            numChanged += self.examineExample(i)
      if(examineAll):
        examineAll = 0
      elif(numChanged == 0):
        examineAll = 1
      self.support_vector()


  def examineExample(self, i2):
    x2 = self.X[i2]
    t2 = self.T[i2]
    y2 = self.y(x2)
    E2 = y2 - t2
    r2 = E2 * t2#KKT条件　0< a_2 < Cのときr2 = 0でないといけない
    a2 = self.a[i2]

    if((0 < a2 or a2 < self.C) and abs(r2) > self.tol):#KKT条件を満たさない時
      lst = [i for i in range(CLASS*N)]
      np.random.shuffle(lst)
      for i1 in lst:
        a1 = self.a[i1]
        if(a1 > 0 and a1 < self.C):
          if(self.takeStep(i1, i2)):
            return 1
      lst = [i for i in range(CLASS*N)]
      np.random.shuffle(lst)
      for i in lst:
        a1 = self.a[i1]
        if(self.takeStep(i1, i2)):
          return 1
    return 0

  def takeStep(self, i1, i2):
    if(i1 == i2):
      return False
    a1_old = self.a[i1]
    a2_old = self.a[i2]
    x1 = self.X[i1]
    x2 = self.X[i2]
    t1 = self.T[i1]
    t2 = self.T[i2]
    y1 = self.y(x1)
    y2 = self.y(x2)
    E1 = y1 - t1
    E2 = y2 - t2

    if(t1 == t2):
      L = max(0,a1_old + a2_old - self.C)
      H = min(self.C, a1_old + a2_old)
    else:
      L = max(0, a2_old - a1_old)
      H = min(self.C, a2_old - a1_old + self.C)
    if(L==H):
      return False

    k11 = self.k(x1,x1)
    k22 = self.k(x2,x2)
    k12 = self.k(x1,x2)
    eta = k11 + k12 - 2*k12
    a2 = a2_old + t2 * (E1 - E2) / eta
    if(a2 < L):
      a2 = L
    elif(a2 > H):
      a2 = H
    if(abs(a2 - a2_old) < self.eps * (a2 + a2_old + self.eps)):
      return False

    a1 = a1_old + t1*t2*(a2_old - a2)
    self.a[i2] = a2
    self.a[i1] = a1
    print(self.cnt,i2,i1,a2_old, a2)
    return True

  def support_vector(self):
    self.sv_index = []
    for i in range(CLASS*N):
      if(self.a[i] > 0):# and self.a[i] < self.C):
        self.sv_index.append(i)

  def decision_bound(self, x):
    ys = np.linspace(-RANGE,RANGE,200)
    mi = 1e10
    mi_i = -1
    for y in ys:
      data = np.array([x,y])
      y_data = abs(self.y(data))
      #print(x,y,y_data)
      if(y_data <= 0.02):
        return y
      if(mi > y_data):
        mi = y_data
        mi_i = y
    return mi_i

  def decision_b(self):
    sm = 0
    cnt = 0
    for n in self.sv_index:
      tn = self.T[n]
      xn = self.X[n]
      sm += tn
      an = self.a[n]
      if(an > 0 and an < self.C):
        self.bound = abs(self.y(xn))
        cnt += 1
      for m in self.sv_index:
        tm = self.T[m]
        am = self.a[m]
        xm = self.X[m]
        knm = self.k(xn,xm)
        sm -= am * tm * knm
    self.b = sm/cnt

phys = []
for i in  range(M):
  phys.append((lambda i: lambda x:pow(x[0],i) + pow(x[1],i))(i))
mean_1 = [11,6]
mean_2 = [-2,-5]
cov_1 = [[100,4], [4, 4]]
cov_2 = [[9,2],[2,16]]

mean_3 = [-3,7]
cov_3 = [[9,0],[0,9]]

cls_1 = []
cls_2 = []
np.random.seed(103)
cls_1.extend(np.random.multivariate_normal(mean_1,cov_1,N))
cls_2.extend(np.random.multivariate_normal(mean_2,cov_2,N))
cls_2.extend(np.random.multivariate_normal(mean_3,cov_3,N))
X = gen_X(cls_1,cls_2)
T = gen_T()

svm = SVM(X,T,"gaussian",loop_max=1e2)
#svm = SVM(X,T,"linear",loop_max=1e3)

xs = np.linspace(-RANGE,RANGE,200)
#ys = [svm.decision_bound(x) for x in xs]


yk = np.linspace(-RANGE,RANGE,200)

Xs, Ys = np.meshgrid(xs,yk)
Z = np.zeros((200,200))
for i in range(200):
  for j in range(200):
    q = np.array([Xs[i][j],Ys[i][j]])
    Z[i][j] += svm.y(q)

pyplot.contour(Xs,Ys,Z,levels=[-1,0,1],colors='m')

#pyplot.plot(xs, ys, label="SVM")

pyplot.scatter(*zip(*cls_1),label="Class1",marker='x',color='r')
pyplot.scatter(*zip(*cls_2),label="Class2",marker='x',color='b')
pyplot.scatter(*zip(*svm.sv),label="support vector",marker='o',facecolors='none', edgecolors='g')
pyplot.legend()
pyplot.show()

