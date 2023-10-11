import sys
sys.path.append("../common")
sys.path.append("../packages/NADP_package")

import pickle
from UT import *
import functions as func
import perturbed_vector as pv
import time
import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import pylab
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import font_manager

print("--Start--")
timeS = time.time()

#############################################

ut = UT()

# -- load

with open("../Data/ebd_mat.pkl", "rb") as f:
    ebd_mat = pickle.load(f)

with open("../Data/top_10.pkl", "rb") as f:
    top_10 = pickle.load(f)

# -- definition of functions

knd = ["N1","N2","N3","N4","N5","J","M","G","L"]
color = {"N1":"black", "N2":"magenta", "N3":"cyan", "N4":"gold", "N5":"blue", "J":"green", "M":"orange", "G":"red", "L":"grey"}
label = {"N1":"NADP (τ=0.10)", "N2":"NADP (τ=0.20)", "N3":"NADP (τ=0.30)", "N4":"NADP (τ=0.40)", "N5":"NADP (τ=0.50)", "J":"Jaccard", "M":"Mahalanobis", "G":"Gaussian", "L":"Laplacian"}
perturbed_vectors_dict = {k : pv.perturbed_vector(k) for k in knd}
parameters = [4*i/5 for i in range(1,51)]
ebd_mat_list = ebd_mat.T.tolist()
n = len(ebd_mat_list)
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(ebd_mat_list)

def jaccard(k,para,i):
    print("skewness",k,para,i)
    perturbed_vectors = perturbed_vectors_dict[k] 
    perturbed_vec = perturbed_vectors[para]
    top_10_former = set(top_10[i])
    try:
        top_10_perturbed = set(neigh.kneighbors([perturbed_vec[i]])[1][0])
    except:
        top_10_perturbed = set(func.top_k_index(np.array(perturbed_vec[i]).T,10,ebd_mat))
    # -- calc Jaccard index
    its, uni = top_10_former & top_10_perturbed, top_10_former | top_10_perturbed
    jac = float(len(its)/len(uni))
    return (para,jac)

def para_skewness(para,para_jac_list):
    jac_list = [x[1] for x in para_jac_list if x[0] == para]
    skewness = ut.skewness(jac_list)
    return (para, skewness)

def jaccard_skewness(k):
    with Pool(32) as pool:
        X = pool.starmap(jaccard,itertools.product([k],parameters,range(0,n)))
    with Pool(32) as pool:
        Y = pool.starmap(para_skewness,itertools.product(parameters,[X]))
    return dict(Y)

def main():
    knd_skewness = dict()
    for k in knd:
        knd_skewness[k] = jaccard_skewness(k)
    return knd_skewness    

# -- execution

if __name__ == "__main__":
    result = main()

print(result)

with open("../Data/skewness.pkl", "wb") as f:
    pickle.dump(result,f)

# -- draw a graph

DIR_root = "../Data/"

plt.rcParams["font.size"] = 27
pylab.figure( num=None, figsize=(13, 13) )
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
plt.subplot(1,1,1)

xlim(left=min(parameters), right=max(parameters));
xlabel("ε",fontsize = 32)
ylim(bottom=-20, top=150);
ylabel("skewness",fontsize = 32)

# -- plot

for k in knd:
    skewness = [result[k][para] for para in parameters]
    plt.plot(parameters,skewness,color=color[k],linewidth=2,linestyle="solid",label=label[k])

plt.xticks(position=(0.0,-0.03),fontsize = 32)
plt.yticks(position=(0.0,-0.03),fontsize = 32)
plt.legend(fontsize=16, loc='best')

# -- save

fle = DIR_root + "skewness" + ".png"
print("Info in method(): save png; fle =", fle )
plt.savefig( fle )
plt.clf()

#####################################
timeE = time.time(); sec = int(timeE-timeS); min = (sec // int(60)) % int(60); hour = sec // int(3600)
print("Time: (" + str(hour) + "h" + str(min) + "m)")

