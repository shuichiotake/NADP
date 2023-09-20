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

with open("/home/otake/benchmark/num_voc_dict_benchmark.pkl", "rb") as f:
    num_voc_10K = pickle.load(f)

with open("/home/otake/benchmark/top_10_data.pkl", "rb") as f:
    top_10_data = pickle.load(f)

n = ebd_mat.shape[1]
A = range(0,n)

data = list()

for i, j in itertools.product(A,A):
    X_i, X_j = set(top_10_data[i]), set(top_10_data[j])
    its, uni = X_i & X_j, X_i | X_j
    jac = float(len(its)/len(uni))
#    if jac >= 0.30:
    if (i in top_10_data[j][0:2] or j in top_10_data[i][0:2]) and (jac >= 0.05):
        dist = np.linalg.norm(ebd_mat[:,i:i+1] - ebd_mat[:,j:j+1])
        data.append((i,j,dist))
        print((i,j,dist))

def sen(i):
    list_i = [(x[1],x[2]) for x in data if x[0] == i]
    sorted_list_i = sorted(list_i,key = lambda x: x[1])
    return (i,sorted_list_i)

#B = list(set([x[0] for x in data]))
#print(len(B))

def main():
    with Pool(12) as pool:
        list_all = pool.map(sen,A)
    return dict(list_all)
        
result = main()

#del result[0][-3:]
#del result[49089][-2:]
#del result[7979][-1]
#del result[69018][-2:]

with open("/data2/otake/benchmark/Data/num_neighbourhoods_0.05.pkl", "wb") as f:
    pickle.dump(result,f)

#####################################
timeE = time.time(); sec = int(timeE-timeS); min = (sec // int(60)) % int(60); hour = sec // int(3600)
print("Time: (" + str(hour) + "h" + str(min) + "m)")




