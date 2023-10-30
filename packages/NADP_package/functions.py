import numpy as np
import math

#############################################      

def top_k(vec,k,ebd_mat):
    n = ebd_mat.shape[1]
    A = ebd_mat - vec.reshape(300,1)
    B = A*A
    C = B.sum(0)
    num_norm_list = list()
    for i in range(0,n):
        num_norm = (i,C[i])
        num_norm_list.append(num_norm)
    T_L = sorted(num_norm_list,key = lambda x: x[1])[0:k]
    L = [x[0] for x in T_L]
    return L

def top_k_index(vec,k,ebd_mat):
    A = ebd_mat - vec.reshape(300,1)
    B = A*A
    C = B.sum(0)
    L = np.argsort(C)[0:k]
    return L

def top_k_norm(vec,k,ebd_mat):
    n = ebd_mat.shape[1]
    A = ebd_mat - vec.reshape(300,1)
    B = A*A
    C = B.sum(0)
    num_norm_list = list()
    for i in range(0,n):
        num_norm = (i,C[i])
        num_norm_list.append(num_norm)
    T_L = sorted(num_norm_list,key = lambda x: x[1])[0:k]
    L = [(x[0],pow(x[1],1/2)) for x in T_L]
    return L

def jac_pre_dist_analytic_gaussian(voc, para, alpha, knd, top_k, vocs_norms):
    #parameters
    l_1 = 1.8356207354
    l_2 = 1.2764672125
    #calc jac
    if knd == 0.10:
        sd_tent = alpha[para]*10.088664096210367
    if knd == 0.15:
        sd_tent = alpha[para]*9.602291597840644
    if knd == 0.20:
        sd_tent = alpha[para]*9.585588802114408
    if knd == 0.30:
        sd_tent = alpha[para]*9.303671968257973
    if knd == 0.40:
        sd_tent = alpha[para]*8.882147129847692
    if knd == 0.50:
        sd_tent = alpha[para]*7.995965407151529
    dist = np.average(np.array(vocs_norms[voc])[1:top_k])
    if dist == 0:
        jac = 0
    elif 0 < dist < 6:
        jac = math.exp(-pow(dist,l_1/2 )*sd_tent)
    else:
        jac = math.exp(-pow(dist,l_2/2 )*sd_tent)
    #calc sd
    if dist == 0:
        sd = sd_tent
    elif 0 < dist < 6:
        sd = -pow(dist,-l_1/2 )*math.log(jac)
    else:
        sd = -pow(dist,-l_2/2 )*math.log(jac)
    return sd

def Mahalanobis(para, sigma_mat):
    X = np.random.normal(0,1,300)
    N = np.reshape(X/np.linalg.norm(X),(300,1))
    Y = np.random.gamma(300,1/para)
    Z = Y*np.dot(sigma_mat,N)
    noise_vec = np.ravel(Z)
    return noise_vec

