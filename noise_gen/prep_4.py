import sys
sys.path.append('/home/machide/WorkSp/PJ_D9W/Body')
sys.path.append('/data1/common/DP')    

from SP import *
from UT_1 import *
import nearest_neighbours_2 as nn
#from p_C import p_C
import os
import time
import itertools
import numpy    as np
from joblib import Parallel, delayed
from multiprocessing import Pool

print("--Start--")
timeS = time.time()
#############################################
ut = UT()

# -- load

with open("/home/otake/benchmark/ebd_mat_benchmark.pkl", "rb") as f:
    ebd_mat = pickle.load(f)

with open("/data2/otake/benchmark/Data/num_neighbourhoods_0.05.pkl", "rb") as f:
    result = pickle.load(f)

with open("/home/otake/benchmark/voc_num_dict_benchmark.pkl", "rb") as f:
    voc_num = pickle.load(f)

#with open("/home/otake/benchmark/num_neighbourhoods_0.10.pkl", "rb") as f:
#    result_sub = pickle.load(f)

with open("/home/otake/benchmark/num_voc_dict_benchmark.pkl", "rb") as f:
    num_voc = pickle.load(f)

#print(result[voc_num["We"]])

#exit()

#for i in range(0,73404):
#     if result[i][-1][1] == 9.303671968257973:
#         print(i)

#exit() 

#del result[0][-3:]
#del result[49089][-2:]
#del result[7979][-1]
#del result[69018][-2:]

#print(result)

#with open("/data2/otake/benchmark/Data/num_neighbourhoods_0.30_mod.pkl", "wb") as f:
#    pickle.dump(f,result)

#exit()

#print(result[0])
#print(result[49089])
#print(result[7979])
#print(result[69018])

#exit()

#result[14615].pop(-1) most important
#result[64435].pop(-1)

#print(result[14615])
#print(result[64435])
#print(result[29979])
#print(result[40517])
#print(result_sub[29979])
#print(result_sub[40517])
#print(result[69426])
#print(result[70188])
#print(num_voc[29979])
#print(num_voc[40517])

#exit()

#print(num_voc[72858],num_voc[73110])
#print(num_voc[71866],num_voc[72589])

#dist_1 = np.linalg.norm(ebd_mat[:,72858:72858+1])
#dist_2 = np.linalg.norm(ebd_mat[:,73110:73110+1])

#dist_3 = np.linalg.norm(ebd_mat[:,71866:71866+1])
#dist_4 = np.linalg.norm(ebd_mat[:,72589:72589+1])

#dist_5 = np.linalg.norm(ebd_mat[:,71866:71866+1]-ebd_mat[:,72589:72589+1])

#print(dist_1,dist_2)
#print(dist_3,dist_4)
#print(dist_5)

#exit()

n = ebd_mat.shape[1]
A = range(0,n)
#A = list(set(A) - {29979})

#r = {29979}
r = set()
s = dict()
t = {0:set()}
u = dict()
i = 1

while True:
    s[i] = dict()
    m = min(set(A) - r)
    z = {0:{m}}
    j = 1
    z[j] = set(z[0])
    for x in result[m]:
        s[i][(m,x[0])] = x[1]
        z[j].add(x[0])
    r = r | z[j]
    j=j+1
    print(len(z[0]),len(z[1]),len(r))
    while len(z[j-1] - z[j-2]) > 0:
        z[j] = set(z[j-1])
        for m in list(z[j-1] - z[j-2]):
            for x in result[m]:
                s[i][(m,x[0])] = x[1]
                z[j].add(x[0])
        r = r | z[j]        
        j=j+1
    t[i] = set(r)    
    if len(r) == n:
        break
    else:
        i=i+1

for j in range(1,i+1):
    u[j] = (t[j] - t[j-1],max(s[j].values()))

#with open("/data2/otake/benchmark/Data/neighbourhood_graph_0.05.pkl", "wb") as f:
#    pickle.dump(s,f)

#with open("/data2/otake/benchmark/Data/Deltas_0.05.pkl", "wb") as f:
#    pickle.dump(u,f)    

#for j in range(1,i+1):
#    print((j,u[j][1],len(u[j][0])))

#####################################
timeE = time.time(); sec = int(timeE-timeS); min = (sec // int(60)) % int(60); hour = sec // int(3600)
print("Time: (" + str(hour) + "h" + str(min) + "m)")




