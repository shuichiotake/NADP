import sys
sys.path.append('/home/machide/WorkSp/PJ_D9W/Body')
sys.path.append('/data1/common/DP')    

from SP import *
from UT_1 import *
from scipy.stats import norm
#from nearest_neighbours_2 import *
#from p_C import p_C
import os
import time
import itertools
import numpy    as np
import math
from joblib import Parallel, delayed
from multiprocessing import Pool

print("--Start--")
timeS = time.time()
#############################################

alpha = dict() 

def Bf(e,u):
    value = norm.cdf(1/(2*u)-e*u)-math.exp(e)*norm.cdf(-1/(2*u)-e*u)
    return value

def B(e):
    u = 1
    while Bf(e,u) >= 1/73404:
        u += 1
    u = u-1
    i = 1
    while Bf(e,u+pow(10,-1)*i) >= 1/73404:
        i += 1
    i = i-1
    u = u + pow(10,-1)*i
    i = 1
    while Bf(e,u+pow(10,-2)*i) >= 1/73404:
        i += 1
    i = i-1
    u = u + pow(10,-2)*i
    i = 1
    while Bf(e,u+pow(10,-3)*i) >= 1/73404:
        i += 1
    i = i-1
    u = u+pow(10,-3)*i
    i = 1
    while Bf(e,u+pow(10,-4)*i) >= 1/73404:
        i += 1
    i = i-1
    u = u+pow(10,-4)*i
    i = 1
    while Bf(e,u+pow(10,-5)*i) >= 1/73404:
        i += 1
    i = i-1
    u = u+pow(10,-5)*i
    i = 1
    while Bf(e,u+pow(10,-6)*i) >= 1/73404:
        i += 1
    i = i-1
    u = u+pow(10,-6)*i
    i = 1
    while Bf(e,u+pow(10,-7)*i) >= 1/73404:
        i += 1
    i = i-1
    u = u+pow(10,-7)*i
    i = 1
    while Bf(e,u+pow(10,-8)*i) >= 1/73404:
        i += 1
    i = i-1
    u = u+pow(10,-8)*i
    i = 1
    while Bf(e,u+pow(10,-9)*i) >= 1/73404:
        i += 1
    i = i-1
    u = u+pow(10,-9)*i
    i = 1
    while Bf(e,u+pow(10,-10)*i) >= 1/73404:
        i += 1
    u = u+pow(10,-10)*i
    return u

for e in [i/10 for i in range(1,401)]:
    alpha[e] = B(e) 

with open("/home/otake/benchmark/Data/alpha_mod.pkl", "wb") as f:
    pickle.dump(alpha,f)

#####################################
timeE = time.time(); sec = int(timeE-timeS); min = (sec // int(60)) % int(60); hour = sec // int(3600)
print("Time: (" + str(hour) + "h" + str(min) + "m)")




