"""
Create the pickle file "wrdKnrms4S.pkl" in the directory "../Data_OM/0_Init/" for preparation
Please try test mode (for which onT = True below) first, because many times are necessary for non-test mode.
"""


import sys
sys.path.append('../../../common/')
sys.path.append("../../Work/")

from UT import *        #from p_C import p_C
from WS import WS

import itertools,numpy,os,pickle,time

print("--Start--")      #H timeS = time.time()
#############################################
ut = UT()
ws = WS(ut)
onT = False     # False
# ----- init
topK = 100   # 100, but 20 is enough
# ----- load
fle = ws._dir_common + "vocKebds_benchmark_TW_73404.pkl"
with open(fle,"rb") as f:     wrdKebds = pickle.load(f)
if onT:
    flg = set(list(wrdKebds.keys())[:1000])
    wrdKebds = { wrd:ebd for wrd,ebd in wrdKebds.items() if wrd in flg }
#:
with open("../Data_OM/9_FromPaper/vocKebds.pkl","rb") as f:     wrdKebds.update( pickle.load(f) )
#:
print("@info: #wrds,topK =", len(wrdKebds),topK)
# ----- create files for norms of neighbourhood of each word
wrdKnrms4S = ws.m0_create_nhood(wrdKebds,topK)          #HH wrdKnrms4S = pAgt.wrdKnrms4S(wrdKebds_mdl,rad,topK)
print("@info: #wrd (in result) =", len(wrdKnrms4S))
# -- dump
fle = ws._dir_OM_0 + "wrdKnrms4S.pkl"
if onT:     fle += "_onT"
with open(fle,"wb") as f:                 pickle.dump(wrdKnrms4S,f,protocol=4)
