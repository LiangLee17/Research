import numpy as np
import scipy as sci
from matplotlib import pyplot as plt
from sklearn.feature_selection import chi2


#some argument: L(boundary), n, h = L/(n+1)
L = 1.0
x0 = -L
y0 = -L
x1 = L
y1 = L
n = 5
h = 2.0*L/(n+1)
idx = 18
y0 + (idx // n+1)*h
x0 + ((idx%n) + 1)*h


#指标全局转换
def gDofIdx2Pnt(x0, y0, n, h, i):
    if i > n**2 - 1 :
        return (-999, -999)
    x = x0 + (i % n + 1) * h
    y = y0 + (i // n + 1) * h
    return x, y

def gEleIdx2PntIdx(n, ei):
    if ei > (n-1)**2 - 1:
        return(-1 , -1, -1, -1)
    LB = ei + ei/(n-1)
    return [LB, LB+1, LB+n+1, LB+n]

def Chi(xi, eta):
    Chi1 = (xi - 1.0)*(eta - 1.0) / 4.0
    Chi2 = -(xi + 1.0)*(eta - 1.0) / 4.0
    Chi3 = (xi + 1.0)*(eta + 1.0) / 4.0
    Chi4 = -(xi - 1.0)*(eta + 1.0) / 4.0
    return [Chi1, Chi2, Chi3, Chi4]

def gradientChi(xi, eta) :
    Chi1xi = (eta - 1.0) / 4.0
    Chi1eta = (xi - 1.0) / 4.0
    Chi2xi = (1.0 - eta) / 4.0
    Chi2eta = -(xi - 1.0) / 4.0
    Chi3xi = (eta + 1.0) / 4.0
    Chi3eta = (xi + 1.0) / 4.0
    Chi4xi = - (eta + 1.0) / 4.0
    Chi4eta = - (xi - 1.0) / 4.0
    return [[Chi1xi, Chi1eta], [Chi2xi, Chi2eta], [Chi3xi, Chi3eta], [Chi4xi, Chi4eta]]


def gEleIdx2Pnt(n, ei) :
    gdi = gEleIdx2PntIdx(n, ei)
    pnts = np.zeros((4, 2), dtype = float)
    k = 0
    for gi in gdi:
        pnts[k][0], pnts[k][1] = gDofIdx2Pnt(x0, y0, n, h, gi)
        k = k+1
    return pnts




