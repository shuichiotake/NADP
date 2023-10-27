import sys
sys.path.append("../packages/NADP_package/")

import pickle
import time
import itertools
import preparation as prep
import numpy as np
import math
from multiprocessing import Pool

print("--Start--")
timeS = time.time()

#############################################

prep.preparation()

# -- load

with open("../noise_gen/data/ebd_mat.pkl", "rb") as f:
    ebd_mat = pickle.load(f)

with open("../noise_gen/data/num_voc_dict.pkl", "rb") as f:
    num_voc = pickle.load(f)

with open("../noise_gen/data/voc_num_dict.pkl", "rb") as f:
    voc_num = pickle.load(f)

with open("../noise_gen/data/top_10.pkl", "rb") as f:
    top_10 = pickle.load(f)

# -- definition of parameters

n = ebd_mat.shape[1]
a = 10
b = 5
lower_bounds = [i/a for i in range(1,b+1)]

m = n // 10000
R = {i : range(10000*i,10000*(i+1)) for i in range(0,m)}
R[m] = range(10000*m,n)

# -- definition of functions

def Jaccard_index(i,j):
    X_i, X_j = set(top_10[i]), set(top_10[j])
    its, uni = X_i & X_j, X_i | X_j
    jac = float(len(its)/len(uni))
    return jac

def neighbourhood(i,j):
    jac = Jaccard_index(i,j)
    # -- round down jac to the second decimal place
    jac_mod = math.floor(10*jac)/10
    if jac_mod > lower_bounds[-1]:
        jac_mod = lower_bounds[-1]
    else:
        pass
    if (i in top_10[j][0:2] or j in top_10[i][0:2]) and (jac_mod >= lower_bounds[0]/2):
        dist = np.linalg.norm(ebd_mat[:,i:i+1] - ebd_mat[:,j:j+1])
        print((i,j,jac_mod,dist))
        return (i,j,jac_mod,dist)
    else:
        pass

def collect(i,k,L):
    X = [(x[1],x[3]) for x in L if (x[2] >= k) and (x[0] == i)]
    X = sorted(X, key = lambda x: x[1])
    return ((i,k),X)

def main(X,Y):
    with Pool(32) as pool:
        Z = pool.starmap(X,Y)
    return Z

L = []
    
for i, j in itertools.product(range(0,m+1),range(0,m+1)):
    domain = itertools.product(R[i],R[j])
    partial_L = main(neighbourhood,domain)
    partial_L = [x for x in partial_L if x != None]
    L += partial_L

edges = {i : dict() for i in lower_bounds}

domain = itertools.product(range(0,n),lower_bounds,[L])
edge_data = main(collect,domain)
edge_data = dict(edge_data)

for i, k in itertools.product(range(0,n),lower_bounds):
    edges[k][i] = edge_data[(i,k)]

# -- adjustment -- If you use your original data, please comment out the following lines befor return.
    
edges[0.1][29979].remove(edges[0.1][29979][-1])
edges[0.1][40517].remove(edges[0.1][40517][-1])
edges[0.2][29979].remove(edges[0.2][29979][-1])
edges[0.2][40517].remove(edges[0.2][40517][-1])
edges[0.4][14615].remove(edges[0.4][14615][-1])
edges[0.4][64435].remove(edges[0.4][64435][-1])
edges[0.5][14615].remove(edges[0.5][14615][-1])
edges[0.5][64435].remove(edges[0.5][64435][-1])

def nbd_graph(k):
    neighbourhoods = edges[k]
    i = 1
    r = set()
    s = dict()
    t = {0:set()}
    u = dict()
    while True:
        s[i] = dict()
        m = min(set(range(0,n)) - r)
        z = {0:{m}}
        j = 1
        z[j] = set(z[0])
        for x in neighbourhoods[m]:
            s[i][(m,x[0])] = x[1]
            z[j].add(x[0])
        r = r | z[j]
        j=j+1
        print(len(z[0]),len(z[1]),len(r))
        while len(z[j-1] - z[j-2]) > 0:
            z[j] = set(z[j-1])
            for m in list(z[j-1] - z[j-2]):
                for x in neighbourhoods[m]:
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
        u[j] = (t[j] - t[j-1], max(s[j].values()))
    return (k,u)

with Pool(32) as pool:
    nbd_graph_list = pool.map(nbd_graph,lower_bounds)
    nbd_graph_dict = dict(nbd_graph_list)

# -- determination of Delta for each vocabullary

Delta_list = list()
s = "voc_Delta_{:.2f}"

def Delta(k):
    nbd_graph = nbd_graph_dict[k]
    n = len(nbd_graph.keys())
    for i in range(1,n+1):
        Delta_list.append(nbd_graph[i][1])
    voc_Delta = dict()
    for i in range(1,n+1):
        for j in list(nbd_graph[i][0]):
            voc = num_voc[j]
            voc_Delta[voc] = nbd_graph[i][1]
    with open("../common/" + str(s.format(k)) + ".pkl", "wb") as f:
        pickle.dump(voc_Delta,f)
    print(len(voc_Delta.keys()))

with Pool(32) as pool:
    pool.map(Delta,lower_bounds)

#####################################
timeE = time.time(); sec = int(timeE-timeS); min = (sec // int(60)) % int(60); hour = sec // int(3600)
print("Time: (" + str(hour) + "h" + str(min) + "m)")

