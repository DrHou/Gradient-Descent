#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:31:59 2018

@author: houzenghao
"""

import numpy as np
import matplotlib.pyplot as plt
import time
Iter = []
Err = []
start = time.time()

m = 500
n = 1000
A = np.random.randn(n,m)
b = np.sign(np.random.random((m,1))-0.5)
#b = np.transpose(np.sign(np.random.random((m,1))-0.5))
one = np.ones(n)
w = np.random.random((n+1,1))
#w = np.zeros((n+1,1)) #variables
#c = np.repeat(w[-1],500)
c = w[-1]
w = w[:n]
ori = np.zeros(m)
orif =0
orinf = 0 
com = np.random.random((n,m))
d = np.zeros((n,1))
ex = 0
t =1 
alpha = 0.1
beta = 0.7

B = np.random.randn(n,m)
for i in range(m):
    B[:,i] = A[:,i]*b[i]

B = np.transpose(B)
b = np.transpose(b)
P = 1/(1+np.exp(-np.dot(B,w)-np.transpose(b*c)))
W = -np.dot(np.transpose(B),(1-P))/m
C = -np.dot(b,(1-P))/m
n = 0     
print "Initialization:", time.time()-start   
a = time.time()
while np.linalg.norm(np.append(W,C),ord=2) > 0.001:
    orif = np.mean(np.log(np.exp(-np.transpose(b)*(np.dot(np.transpose(A),w)+c))+1))
    orinf = np.mean(np.log(np.exp(-np.transpose(b)*(np.dot(np.transpose(A),w - W*t)+c-C*t))+1))
    print time.time()-a   
    right = orif - alpha*t*(np.linalg.norm(np.append(W,C),ord=2))**2                          
    while orinf >= right:
        t = t*beta
        orinf = np.mean(np.log(np.exp(-np.transpose(b)*(np.dot(np.transpose(A),w - W*t)+c-C*t))+1))     
        right = orif - alpha*t*(np.linalg.norm(np.append(W,C),ord=2))**2
    
    print np.linalg.norm(np.append(W,C),ord=2)
    print n
    w = w-t*W
    c = c-t*C
    P = 1/(1+np.exp(-np.dot(B,w)-np.transpose(b*c)))
    W = -np.dot(np.transpose(B),(1-P))/m
    C = -np.dot(b,(1-P))/m
    n = n+1
    Iter.append(np.log10(np.linalg.norm(np.append(W,C),ord=2)))
    Err.append(n)
print "Finish computation", time.time()-start
#-------------- plotting -------------------

plt.clf()
plt.plot(Err, Iter, marker='o', markersize= 2, color='r', label='Eulidean norm of the gradient')
plt.xlabel('No. of Iteration')
plt.ylabel('Gradient norm log10 converted')
plt.title('Logistic regression')
plt.legend()
plt.show()
print "final result:",time.time()-start
'''                               
plt.clf()
plt.plot(Err, Iter, marker='o',linestyle='--', markersize= 4, color='r', label='epsilon')
plt.xlabel('No. of Iteration')
plt.ylabel('Gradient norm log10(')
plt.ylim(0,0.001)
plt.title('Logistic regression')
plt.legend()
plt.show()   
'''                 
