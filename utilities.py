import math
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange, float64, int64

@njit(nogil=True)
def φ_n(z, l) -> np.array:
    x1, x2 = 0, 6
    return  4*16 * (np.sin((x2) * 2 * z / l) - np.sin((x1) * 2 * z / l)) / (2 * z + np.sin(2 * z))


@njit(nogil=True)
def half_method(n: int, a1: float, b1: float, eps, l, c, α) -> np.ndarray:
    z = list()

    def find_c(a: float, b: float) -> float:
        G = α / (c * l)
        fz = lambda z: (np.tan(z) - G / z)
        root = (a + b) / 2
        while np.abs(a - b) > eps:
            if (fz(root) * fz(a)) < 0:
                b = root
            else:
                a = root
            root = (a + b) / 2
        return root

    while n > 0:
        root = find_c(a1, b1)
        z.append(root)
        a1 += np.pi
        b1 += np.pi
        n -= 1
    return np.array(z)


@njit(nogil=True)
def w_n(z, φ, time, c, k, R, l, α, x_list, x, time_list, flag='x'):
    def integralfi(φi, zi, l):
        return φi*l*(1+np.sin(2*zi)/(2*zi))/4
    def P_n(zi, φi, t):
        a2 = (k / c) ** 2
        aRc = (2 * α / R) / (c ** 2)
        return (φi * 4 * (1 - np.exp(-1 * ((a2 * 4 * (zi ** 2) / (l ** 2) + aRc) * t))) / (
                a2 * 4 * (zi ** 2) / (l ** 2) + aRc)) / (c * l)

    def w(z, φ, ti, x):
        s = list()
        for i in prange(len(z)):
            s.append(P_n(z[i], φ[i], ti) * np.cos(((2 * z[i] / l) * (x))))

        return np.sum(np.array(s))

    sol = []
    if flag == 'x':
        for i in x_list:
            sol.append(w(z, φ, time, i))
    elif flag == 't':
        for i in time_list:
            sol.append(w(z, φ, i, x))

    return sol


@njit(nogil=True)
def solutions(n, t, α, c, l, k, R, eps, x_list, time_list, x, flag='x'):
    z = half_method(n, 0.000001, np.pi / 2, eps, l, c, α)
    φ = φ_n(z, l)

    solution = w_n(z=z, φ=φ, α=α, c=c, l=l, k=k, R=R, time=t, flag=flag, x=x, x_list=x_list, time_list=time_list)

    return solution, z, φ

@njit(nogil=True, cache=True)
def w_l2T(z, φ, time, x, α, c, l, T, k, R):
    def P_n(zi, φi, t):
        a2 = (k / c) ** 2
        aRc = (2 * α / R) / (c ** 2)

        return (φi * 4 * (1 - np.exp(-1 * ((a2 * 4 * (zi ** 2) / (l ** 2) + aRc) * t))) / (
                a2 * 4 * (zi ** 2) / (l ** 2) + aRc)) / (c * l)

    def w(z, φ, ti, x):
        s = list()
        for i in prange(len(z)):
            s.append(P_n(z[i], φ[i], ti) * np.cos(((2 * z[i] / l) * (x))))

        return np.sum(np.array(s))

    sol = w(z, φ, time, x)

    return sol

@njit(float64(float64, int64))
def truncate(f, accuracy):
    return math.floor(f * 10 ** accuracy) / 10 ** accuracy

def solution_analytic(n, α, c, l, T, k, R, eps):
    hx = 1 / 10
    x_list = np.arange(0.0, l + hx, hx)
    ht = 1 / 10
    time_list = np.arange(0.0, T + ht, ht)

    t = 5
    numOfThreads = 8
    results_x = {}
    results_t = {}

    for i in prange(numOfThreads):
        [results_x[(t + 35 * i)], z1, φ1] = solutions(n=n, t=t + 35 * i, α=α, c=c, l=l, k=k, R=R, eps=eps,
                                                      x=l / 2,
                                                      flag='x',
                                                      x_list=x_list, time_list=time_list)
    for i in prange(numOfThreads - 1):
        [results_t[i], z1, φ1] = solutions(n=n, t=T, α=α, c=c, l=l, k=k, R=R, eps=eps, flag='t', x=i, x_list=x_list,
                                           time_list=time_list)
    return results_x, x_list, results_t, time_list, z1, φ1

@njit(nogil=True, cache=True)
def Fi(u0k, xi, c, ht):
    return u0k*c/ht+fix(xi)

@njit(nogil=True, cache=True)
def fix(x):
    if -2<=x<=2:
        return 16
    return 0

@njit(nogil=True, cache=True)
def betta0(u0k, xi, c, ht, B0):
    return Fi(u0k, xi, c, ht)/B0

@njit(nogil=True, cache=True)
def alfai(alfai_1, Ci, Ai, Bi):
    return -Ci/(Ai*alfai_1 + Bi) 

@njit(nogil=True, cache=True)
def bettai(betta_1, alfa_1, Fi, Ai, Bi):
    return (Fi - Ai*betta_1)/(Ai*alfa_1 + Bi)

@njit(nogil=True, cache=True)
def Wik1(bettaI_1, alfaI_1, FI, AI, BI):
    return (FI - bettaI_1*AI)/(AI*alfaI_1+BI)
   
@njit(nogil=True, cache=True)
def solution_implicit(I, K, α, c, l, T, k, R, eps):
    I = I + 1
    K = K + 1
    x_list = np.linspace(0.0, l, I)
    hx = x_list[1] - x_list[0]

    time_list = np.linspace(0.0, T, K)
    ht = time_list[1] - time_list[0]

    u = np.zeros((I+1, K+1))

    ## Constants:
    B0 = c/ht+2*k/(hx**2) + 2*α/R/c
    C0 = -2*k/(hx**2)
    Ai = -k/(hx**2)
    Bi = c/ht + 2*k/(hx**2) + 2*α/R/c
    Ci = -k/(hx**2)

    AI = -2*k/(hx**2)
    BI = c/ht+2*k/(hx**2) + 2*α/R/c + 2*α*k/c/hx
    alfa0 = -C0/B0

    for k in range(0, K):
        alfa, betta = list(), list()
        alfa.append(alfa0)
        betta.append(betta0(u[0][k], x_list[0], c, ht, B0))

        i = 2
        while i < I: 
            alfa.append(alfai(alfa[i-2], Ci, Ai, Bi))
            betta.append(bettai(betta[i-2], alfa[i-2], Fi(u[i][k], x_list[i], c, ht), Ai, Bi))
            i+=1
        
        u[i][k+1] = Wik1(betta[i-2], alfa[i-2], Fi(u[i - 2][k], x_list[i-2], c, ht), AI, BI)

        i-=1    
        while i>=1:
            u[i][k+1] = alfa[i-1]*u[i+1][k+1] + betta[i-1]
            i -= 1
        
        u[0][k+1] = alfa0*u[1][k+1] + betta[0]
        u[1][k+1] = u[0][k+1]
             
    return x_list, time_list, u, ht, hx
