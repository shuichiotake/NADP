import sys
sys.path.append('/home/machide/WorkSp/PJ_D9W/Body')     

from SP import *
#from p_C import p_C
import os
import time
#import itertools
from itertools import chain, combinations
import numpy    as np
import csv
from multiprocessing import Pool

print("--Start--")
timeS = time.time()
#############################################      

#load

with open("/data1/common/DP/vocKebds_benchmark_TW_73404.pkl", "rb") as f:
    vocs_ebd_dict = pickle.load(f)

vocs_ebd_list = list(vocs_ebd_dict.items())

n = len(vocs_ebd_list)

num_voc_dict = dict()
voc_num_dict = dict()
num_vec_list = list() 

for i in range(0,n):
    voc = vocs_ebd_list[i][0]
    vec = vocs_ebd_list[i][1]
    num_voc_dict[i] = voc
    voc_num_dict[voc] = i
    num_vec_list.append(vec)

W = np.array(num_vec_list).T

#with open("/home/otake/benchmark/num_voc_dict_benchmark.pkl","wb") as f:
#    pickle.dump(num_voc_dict,f)

#with open("/home/otake/benchmark/voc_num_dict_benchmark.pkl","wb") as f:
#    pickle.dump(voc_num_dict,f)

#with open("/home/otake/benchmark/ebd_mat_benchmark.pkl","wb") as f:
#    pickle.dump(W,f,protocol=4)

print(num_voc_dict[100])
print(voc_num_dict[num_voc_dict[100]])
print(W.shape)

#####################################
timeE = time.time(); sec = int(timeE-timeS); min = (sec // int(60)) % int(60); hour = sec // int(3600)
print("Time: (" + str(hour) + "h" + str(min) + "m)")








