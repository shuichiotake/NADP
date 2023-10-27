import pickle
import itertools
import functions as func
import numpy as np
import scipy.linalg
from scipy.stats import norm
import math
from multiprocessing import Pool

#############################################      

# -- load

with open("../common/vocKebds_benchmark_TW_73404.pkl", "rb") as f:
    vocs_ebd_dict_73404 = pickle.load(f)

with open("../common/vocs_ebd_dict_10593.pkl", "rb") as f:
    vocs_ebd_dict_10593 = pickle.load(f)

# --definition of functions

def get_top_k_norm(i,ebd_mat):
    vec = ebd_mat[:,i:i+1]
    top_k_norm = func.top_k_norm(vec,10,ebd_mat)
    print((i,top_k_norm))
    return (i,top_k_norm)

def preparation():
    num_voc_dict = dict()
    voc_num_dict = dict()
    num_vec_list = list() 
    vocs_ebd_list = list(vocs_ebd_dict_73404.items())
    n = len(vocs_ebd_list)
    for i in range(0,n):
        voc = vocs_ebd_list[i][0]
        vec = vocs_ebd_list[i][1]
        num_voc_dict[i] = voc
        voc_num_dict[voc] = i
        num_vec_list.append(vec)
    W = np.array(num_vec_list).T
    with open("../noise_gen/data/num_voc_dict.pkl","wb") as f:
        pickle.dump(num_voc_dict,f)
    with open("../noise_gen/data/voc_num_dict.pkl","wb") as f:
        pickle.dump(voc_num_dict,f)
    with open("../noise_gen/data/ebd_mat.pkl","wb") as f:
        pickle.dump(W,f,protocol=4)
    domain = itertools.product(range(0,n),[W])
    with Pool(32) as pool:
        top_k_tuples = pool.starmap(get_top_k_norm,domain)
        top_k_dict = dict(top_k_tuples)
        top_k_index_dict = {i:[x[0] for x in top_k_dict[i]] for i in range(0,n)}
        top_k_norm_dict = {num_voc_dict[i]:[x[1] for x in top_k_dict[i]] for i in range(0,n)}
    with open("../noise_gen/data/top_10.pkl", "wb") as f:
        pickle.dump(top_k_index_dict,f)
    with open("../noise_gen/data/top_10_norm.pkl", "wb") as f:
        pickle.dump(top_k_norm_dict,f)
    W = np.array(list(vocs_ebd_dict_10593.values())).T
    s_c_mat = np.cov(W)
    s_s_c_mat = (300/np.trace(s_c_mat))*s_c_mat
    M = scipy.linalg.sqrtm(s_s_c_mat)
    with open("../common/sigma_mat.pkl", "wb") as f:
        pickle.dump(M,f)
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
    with open("../common/alpha_mod.pkl", "wb") as f:
        pickle.dump(alpha,f)

#####################################








