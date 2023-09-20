import sys
sys.path.append('/home/machide/WorkSp/PJ_D9W/Body')    

from SP import *
#from p_C import p_C
import nearest_neighbours_2 as nn
import functions as func
import os
import time
import itertools
import numpy    as np
from joblib import Parallel, delayed
from multiprocessing import Pool

print("--Start--")
timeS = time.time()
#############################################
       
# -- load

with open("/home/otake/benchmark/ebd_mat_benchmark.pkl","rb") as f:
    ebd_mat = pickle.load(f)

with open("/home/otake/benchmark/num_voc_dict_benchmark.pkl","rb") as f:
    num_voc_dict = pickle.load(f)

#n = 51228
n = ebd_mat.shape[1]
#n = 73404

def get_top_k(i):
    vec = ebd_mat[:,i:i+1]
    top_k_list = nn.top_k(vec,10,ebd_mat)
    print(top_k_list)
#    dist = float(sum(func.top_k_norm(vec,10,ebd_mat)[1:])/9)
    print((i,top_k_list))
    return (i,top_k_list)
        
def main():
    with Pool(1) as pool:
        num_top_k_tuples = pool.map(get_top_k,range(0,n))
        num_top_k_dict = dict(num_top_k_tuples)
    return num_top_k_dict
    
if __name__ == "__main__":
    result = main()

#with open("/home/otake/benchmark/top_10_data_mod.pkl", "wb") as f:
#    pickle.dump(result,f)

#for i in range(0,10):
#    print(result[i])

#####################################
timeE = time.time(); sec = int(timeE-timeS); min = (sec // int(60)) % int(60); hour = sec // int(3600)
print("Time: (" + str(hour) + "h" + str(min) + "m)")


