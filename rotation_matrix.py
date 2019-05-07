#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:03:42 2018

@author: zengyang
"""

# Rotation matrix of high-dimensional space
# Refer 1: @article{aguilera2004general,title={General n-dimensional rotations},
#  author={Aguilera, Antonio and P{\'e}rez-Aguila, Ricardo},
#  year={2004},
#  publisher={V{\'a}clav Skala-UNION Agency}
#}
import numpy as np


def rotation_matrix(V, theta):
    '''
    Aguilera-Perez Algorithm
    rotate matrix for high dimensional space
    -------------------------------------------
    input:
    V : array(n-1, n), n is the dimension, 
    the origin of rotation, point in 2D, line in 3D, plane in 4D
    
    theta: the angel of rotation around the origin
    -------------------------------------------
    output
    M: array(n+1, n+1), 
    x' = x * M
    x is the original point
    x' is the point rotated by M
    '''
    if V.shape[1]-V.shape[0] != 1:
        print('Error input')
    
    # expend matrix for calculate T
    V0 = np.concatenate((V, np.ones([V.shape[0], 1])), axis=1)
    
    # calculate T according to Eq.(1)
    M = T(V)
    v = np.dot(V0, M)
    k = 1
    for r in range(1, V.shape[1]-1):
        for c in range(V.shape[1]-1 , r-1, -1):
            k += 1
            Mk = R(theta=arctan2(y=v[r, c], x=v[r, c-1]),
                   n=v.shape[1], a=c, b=c-1)
            v = np.dot(v, Mk)
            M = np.dot(M, Mk)
    n = V.shape[1]
    M1 = np.dot(M, R(theta=theta, n=n+1, a=n-2, b=n-1))
    M2 = np.linalg.inv(M)
    M = np.dot(M1, M2)
    return M    

def T(V):
    '''
    T matrix
    according to Eq.(1) Refer 1.
    '''
    T = np.eye(V.shape[1]+1)
    T[-1] = T[-1] - np.concatenate((V[0], np.zeros(1)), )
    return T

def arctan2(y, x):
    '''
    This function is to calculate arctan2
    '''
    if x > 0:
        theta = np.arctan(y/x)
    elif x < 0:
        theta = np.arctan(y/x) + np.pi
    elif x == 0:
        if y >= 0:
            theta = np.pi/2
        else:
            theta = -np.pi/2
    return theta

def R(theta, n, a, b):
    '''
    Calculate R_{a,b}(theta)
    '''
    Rc = np.eye(n)
    Rc[a, a] = np.cos(theta)
    Rc[b, b] = np.cos(theta)
    Rc[a, b] = np.sin(theta)
    Rc[b, a] = - np.sin(theta)
    return Rc

# Test the Algorithm

##################################
#    2D
#V = np.array([[1,1]])
#V_r = np.array([[2, 1, 1]])
#theta = np.pi/2
#M = rotation_matrix(V, theta)
#print(np.dot(V_r, M))



##################################
#    3D
# V = np.array([[0,1,1],[2, 1, 1]])
# theta = np.pi/2
# Mr = rotation_matrix(V, theta)
# Vt = np.array([[0, 2, 1, 1]])
# print(np.dot(Vt, Mr))
    



    
