import sys
sys.path.append("../common/")

import pickle
from UT import *
import functions as func
import time
import itertools
import numpy as np
from multiprocessing import Pool

#############################################

ut = UT()

# -- load

with open("../Data/ebd_mat.pkl", "rb") as f:
    ebd_mat = pickle.load(f)

with open("../Data/num_voc_dict.pkl", "rb") as f:
    num_voc_dict = pickle.load(f)

with open("../Data/top_10_norm.pkl", "rb") as f:
    top_10_norm = pickle.load(f)

with open("../Data/alpha_mod.pkl", "rb") as f:
    alpha = pickle.load(f)

with open("../Data/sigma_mat.pkl", "rb") as f:
    sigma_mat = pickle.load(f)

with open("../Data/voc_Delta_0.10.pkl", "rb") as f:
    voc_Delta_1 = pickle.load(f)

with open("../Data/voc_Delta_0.20.pkl", "rb") as f:
    voc_Delta_2 = pickle.load(f)

with open("../Data/voc_Delta_0.30.pkl", "rb") as f:
    voc_Delta_3 = pickle.load(f)

with open("../Data/voc_Delta_0.40.pkl", "rb") as f:
    voc_Delta_4 = pickle.load(f)

with open("../Data/voc_Delta_0.50.pkl", "rb") as f:
    voc_Delta_5 = pickle.load(f)

# -- definition of noise_generattion functions

knd = ["N1","N2","N3","N4","N5","J","M","G","L"]
parameters = [4*i/5 for i in range(1,51)]
residue = {para : round(5*para/4) for para in parameters}
n = ebd_mat.shape[1]

def perturb(k,para,i):
    print("perturb",k,para,i)
    if k == "N1":
        np.random.seed(9*(50*i + residue[para]))
        voc = num_voc_dict[i]
        voc_Delta = voc_Delta_1
        sd = ut.analytic_gaussian_var(voc, para, alpha, voc_Delta)
        sd = max(sd,0.00000000000000000001)
        per_vec = np.random.normal(0,sd,300)
    if k == "N2":
        np.random.seed(9*(50*i + residue[para])+1)
        voc = num_voc_dict[i]
        voc_Delta = voc_Delta_2
        sd = ut.analytic_gaussian_var(voc, para, alpha, voc_Delta)
        sd = max(sd,0.00000000000000000001)
        per_vec = np.random.normal(0,sd,300)
    if k == "N3":
        np.random.seed(9*(50*i + residue[para])+2)
        voc = num_voc_dict[i]
        voc_Delta = voc_Delta_3
        sd = ut.analytic_gaussian_var(voc, para, alpha, voc_Delta)
        sd = max(sd,0.00000000000000000001)
        per_vec = np.random.normal(0,sd,300)
    if k == "N4":
        np.random.seed(9*(50*i + residue[para])+3)
        voc = num_voc_dict[i]
        voc_Delta = voc_Delta_4
        sd = ut.analytic_gaussian_var(voc, para, alpha, voc_Delta)
        sd = max(sd,0.00000000000000000001)
        per_vec = np.random.normal(0,sd,300)
    if k == "N5":
        np.random.seed(9*(50*i + residue[para])+4)
        voc = num_voc_dict[i]
        voc_Delta = voc_Delta_5
        sd = ut.analytic_gaussian_var(voc, para, alpha, voc_Delta)
        sd = max(sd,0.00000000000000000001)
        per_vec = np.random.normal(0,sd,300)
    if k == "J":
        np.random.seed(9*(50*i + residue[para])+5)
        voc = num_voc_dict[i]
        sd = func.jac_pre_dist_analytic_gaussian(voc, para, alpha, 0.50, 10, top_10_norm)
        sd = max(sd,0.00000000000000000001)
        per_vec = np.random.normal(0,sd,300)
    if k == "G":
        np.random.seed(9*(50*i + residue[para])+6)
        sd = ut.analytic_gaussian(para,alpha,0.50)
        sd = max(sd,0.00000000000000000001)
        per_vec = np.random.normal(0,sd,300)
    if k == "L":
        np.random.seed(9*(50*i + residue[para])+7)
        per_vec = np.random.laplace(0,151.7991981/para,300)
    if k == "M":
        np.random.seed(9*(50*i + residue[para])+8)
        per_vec = ut.Mahalanobis(para,sigma_mat)
    return ((para, i), per_vec)

def summarize(W, para):
    per_vec_list = [W[(para, i)] for i in range(0,n)]
    per_vec_mat = np.array(per_vec_list).T
    per_mat = ebd_mat + per_vec_mat
    return per_mat.T.tolist()

def main(X, Y):
    with Pool(32) as pool:
        Z = pool.starmap(X, Y)
    return dict(Z)

def perturbed_vector(k):
    per_vec_dict = dict()
    domain = itertools.product([k], parameters, range(0,n))
    result = main(perturb, domain)
    for para in parameters:
        per_vec_dict[para] = summarize(result, para)
    return per_vec_dict

#####################################




