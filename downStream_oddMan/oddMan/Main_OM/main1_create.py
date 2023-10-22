"""
Create pickle files for perturbate word embddings (pebd) from the embdding file "vocKebds_benchmark_TW_73404.pkl" with additional data for oddman task in the common directory.
Please try test mode (for which onT = True below) first; when non-test mode, we need about 3days and 1.5T in HD. 
The following files are necessary before executing this program:
- alpha_mod.pkl
- sigma_mat.pkl                    
- voc_Delta_0.10.pkl  
- voc_Delta_0.50.pkl               
- vocKebds_benchmark_TW_73404.pkl 
- UT.py
"""


import sys
sys.path.append('../../../common/')
sys.path.append("../../Work/")

from UT import *        #from p_C import p_C
from WS import WS

import itertools,numpy,os,time

print("--Start--")      #
timeS = time.time()
#############################################
ut = UT()
ws = WS(ut)
onT = True     # use True for test mode
# ----- load
fle = "../Data_OM/9_FromPaper/vocKebds.pkl"    #fle = ws._dir_common + "vocKebds_benchmark_TW_73404.pkl"
with open(fle,"rb") as f:    wrdKebds = pickle.load(f)
#:
fle = ws._dir_OM_0 + "wrdKnrms4S.pkl"
if onT:     fle += "_onT"
with open(fle,"rb") as f:                wrdKnrms4S = pickle.load(f)       
print("info: #wrds(ebd,nrm),onT =", len(wrdKebds),len(wrdKnrms4S),onT)   # len(wrdKnrms4S)
# ----- set 
knds = ["AGv","AJ","M","AG","AL"]   # kinds of noise: AGv->NADP, AJ->jaccard, M->Maharanobis, AG->Gaussian, AL->Laplacian
jacs = ["0.50"]     #["0.10","0.50"]              # thresholds tau used in the neighbouring relation in Jaccard noise
epss = sorted([ float(i)/10  for i in range(8,401,8) ], reverse=True )      # epsilon-values in (epsilon,delta)-Differential Privacy
lbls = [ "L"+str(i)  for i in range(5) ]    # used for error bar
#:
if onT:     jacs,epss,lbls = jacs[:1],epss[:1],lbls[:1]
# ----- create temporally files 
print("create temporally files ...")
ws.m1_create_NG(wrdKebds,jacs,epss,lbls,onT,typ="OM");print(" NADP,Gaussian done")   # N->NADP,G->Gaussian
print("EEEE");exit()
ws.m1_create_J(wrdKebds,jacs,epss,lbls,wrdKnrms4S,onT,typ="OM");print(" Jaccard done")    # J->Jaccard
ws.m1_create_L(wrdKebds,epss,lbls,onT,typ="OM");print(" Laplacian done")    # L->Laplacian
#ws.m1_create_M(wrdKebds,epss,lbls,onT,typ="OM");print(" Maharanobis done")    # M->Maharanobis
timeE = time.time(); sec = int(timeE-timeS); min = (sec // int(60)) % int(60); hour = sec // int(3600);     print("Current Time: (" + str(hour) + "h" + str(min) + "m)")
# ----- create CWA files
print("create CWA files ...")
ws.create_OM_NGJLM(knds,jacs,epss,lbls,onT);print(" NADP,Gaussian,Jaccard,Laplacian,Maharanobis done")   # N->NADP,G->Gaussian
#####################################
timeE = time.time(); sec = int(timeE-timeS); min = (sec // int(60)) % int(60); hour = sec // int(3600);     print("Total Time: (" + str(hour) + "h" + str(min) + "m)")
