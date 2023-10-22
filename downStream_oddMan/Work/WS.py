"""
This file gives workspaces for main files in Main_DS.
"""

# import
#:main modules
from multiprocessing import Process, Manager
import numpy
#:meta modules and one line
import time,itertools,random,pickle



class WS:
    # ------
    def __init__(self, ut):
        self._ut = ut   # object of UT
        # -----
        self._dimGlv = 300      # dim of glove
        self._dir_common = "../../../common/"      # dir of common
        self._dir_DS_0  = "../Data_DS/0_Init/" 
        self._dir_DS_1  = "../Data_DS/1_VocKpebd/" 
        self._dir_DS_Mdl  = "../Data_DS/Models/" 
        self._dir_OM_0  = "../Data_OM/0_Init/" 
        self._dir_OM_1  = "../Data_OM/1_VocKpebd/" 
        self._dir_OM_9  = "../Data_OM/9_FromPaper/" 
        self._dir_OM_CWA  = "../Data_OM/CWAs/" 
        #?self._dir_OM_  = "../Data_OM/0_VocKpebd/" 
        self._kaomojis = ["(-_-)","(0_0)","(^_^)","(._.)","(@_@)","(-_-)","(^o^)","(^o^)","(;_;*)","(*;_;)","(\'-\'*)","(;^_^A"]

    # ----- for main0_init.py       #HH def wrdKnrms4S(self, vocs_ebd, topK):
    def m0_create_nhood(self, vocs_ebd, topK):
        vocs,ebds = list(vocs_ebd.keys()),list(vocs_ebd.values())
        # ----- make argsZ
        Prcs  = 12 # 15
        siz   = int(len(vocs)/Prcs)
        argsS = [ [{ voc:vocs_ebd[voc] for voc in vocs[siz*i:siz*(i+1)] },ebds,topK] for i in range(Prcs+1) ]
        # ----- cal
        wrdKnrms4S = self.mult_any(argsS = argsS, target = self.wrdKnrms4S_target, retType = "dict")
        # - return
        return wrdKnrms4S
    #:
    def wrdKnrms4S_target(self, rslts, args, nth, cnt):
        vocs_ebd,ebds,topK = args
        #
        tmeS = time.time()
        fnc = random.sample(self._kaomojis,1)[0]
        #
        wrdKnrms4S = { voc:self.help_norms4(vocs_ebd[voc],ebds,topK) for voc in vocs_ebd.keys() }
        #
        rslts.update(wrdKnrms4S)
        # INFO
        ttl = int(time.time()-tmeS); h = ttl // int(3600); m = (ttl // int(60)) % int(60);  s = ttl % 60;
        #DD print( fnc+":nth"+str(nth)+"["+str(cnt.value+1)+"]; len(wrdKnrms4S)="+str(len(wrdKnrms4S))+  " ("+str(h)+"h"+str(m)+"m"+str(s)+"s),", end=" ")
        cnt.value += 1
    #:
    def help_norms4(self, vec, ebds, topK):
        norms = [ numpy.linalg.norm(vec - ebd) for ebd in ebds ]
        return sorted(norms)[:topK]


    # ----- for main1_create_vocKpebds.py
    def m1_create_NG(self, wrdKebds, jacs, epss, lbls, onT, typ=None):
        Dir = self._dir_DS_1 if typ == "DS" else (self._dir_OM_1 if typ == "OM" else None)
        # -----
        knds = ["AGv","AG"]     # AG->NADP, AG->Gaussian
        # ----- init
        with open(self._dir_common+"alpha_mod.pkl","rb") as f:           alpha_mod = pickle.load(f)
        #:
        jacKvoc_Delta = dict()
        for jac in jacs:
            with open(self._dir_common+"voc_Delta_"+jac+".pkl","rb") as f:      voc_Delta = pickle.load(f)
            jacKvoc_Delta[jac] = voc_Delta
        # ----- loop
        for lbl,knd,jac,eps in itertools.product(lbls,knds,jacs,epss):
            # --- pre
            voc_Delta = jacKvoc_Delta[jac] if knd == "AGv" else dict()
            #DD # - info    #DD print("SS: onT,knd,jac,eps,alp,lbl,len(voc_Delta) =", onT,knd,jac,eps,alpha_mod[eps],lbl,len(voc_Delta))
            # - set wrdKpbd, wrdKnbd9nrdsS   # pbd->perturbated ebd, nbd->near ebd, nrd->near wrd
            wrdKpbds = dict()
            for cnt0,(wrd,ebd) in enumerate(sorted(wrdKebds.items())):
                # get noi; use normal Delta value if wrd not in voc_Delta; e.g. Delta ~ 8.00 if jac = 0.50 (see UT.py for exact values)
                noi = self._ut.analytic_gaussian(eps,alpha_mod,float(jac)) if knd == "AG" or wrd not in voc_Delta else \
                      self._ut.analytic_gaussian_var(wrd,eps,alpha_mod,voc_Delta)
                # get pbd
                pbd = self.pbd(ebd,"G",noi)   
                #DD if np.any(np.isnan(pbd)):       print("@@Bug?:",cnt0,wrd,ebd,pbd);  exit()
                # set
                wrdKpbds[wrd] = pbd
            # - dump
            tal = "_".join([str(knd),str(jac),str(eps),str(lbl)])
            mdl = "Noi_" + knd + "/"
            fle = Dir + mdl + "wrdKpbds_" + tal + ".pkl"
            if onT: fle += "_onT"
            with open(fle,"wb") as f:              pickle.dump(wrdKpbds,f)
    #:
    def m1_create_J(self, wrdKebds, jacs, epss, lbls, wrdKnrms4S, onT, typ=None):
        Dir = self._dir_DS_1 if typ == "DS" else (self._dir_OM_1 if typ == "OM" else None)
        knd = "AJ"
        # ----- init
        topK = 10
        with open(self._dir_common+"alpha_mod.pkl","rb") as f:           alpha_mod = pickle.load(f)
        # ----- loop
        for lbl,jac,eps in itertools.product(lbls,jacs,epss):
            # - info
            # - set wrdKpbd, wrdKnbd9nrdsS   # pbd->perturbated ebd, nbd->near ebd, nrd->near wrd
            wrdKpbds = dict()
            for cnt0,(wrd,ebd) in enumerate(sorted(wrdKebds.items())):
                # get noi;  # note; float(jac)<->knd in otake-san
                if sum(wrdKnrms4S[wrd][1:topK]) > 0:
                    noi = self._ut.jac_pre_dist_analytic_gaussian(    wrd,eps,alpha_mod,float(jac),topK,wrdKnrms4S) 
                else:
                    noi = 0.0
                noi = max(noi,0.0)#? max(noi,0.00000000000000000001)   # otake-san kara; riyu ha wakran
                # get pbd
                pbd = self.pbd(ebd,"G",noi)   #machigatta;kirokutoshite nokosu;   pbd = ebd + noi
                #DD if np.any(np.isnan(pbd)):       print("@@Bug?:",cnt0,wrd,ebd,pbd);  print(noi);exit()
                # set
                wrdKpbds[wrd] = pbd
            # - dump
            tal = "_".join([str(knd),str(jac),str(eps),str(lbl)])
            mdl = "Noi_" + knd + "/"
            fle = Dir + mdl + "wrdKpbds_" + tal + ".pkl"    #HH fle = self._dir_DS_1 + mdl + "wrdKpbds_" + tal + ".pkl"
            if onT: fle += "_onT"
            with open(fle,"wb") as f:              pickle.dump(wrdKpbds,f)
    #:
    def m1_create_L(self, wrdKebds, epss, lbls, onT, typ=None):
        Dir = self._dir_DS_1 if typ == "DS" else (self._dir_OM_1 if typ == "OM" else None)
        knd = "AL"
        # -----
        for eps,lbl in itertools.product(epss,lbls):
            #DD print("@@: onT,knd,eps,lbl =", onT,knd,eps,lbl)
            # - set wrdKpbd, wrdKnbd9nrdsS   # pbd->perturbated ebd, nbd->near ebd, nrd->near wrd
            wrdKpbds = dict()
            for cnt0,(wrd,ebd) in enumerate(sorted(wrdKebds.items())):
                # get noi
                noi = self._ut.analytic_laplace(eps)
                # cal
                pbd = self.pbd(ebd,"L",noi)
                #DD if np.any(np.isnan(pbd)):       print("@@Bug?:",cnt0,wrd,ebd,pbd);  exit()
                # set
                wrdKpbds[wrd] = pbd
                #DD # info
                #DD if random.choice(range(10000)) == 0 or cnt0%10000 == 0:      #H if cnt0% 100 == 0:
                #DD     print("@@@: (onT,knd,eps,noi),(cnt0,wrd),ebd[:2],pbd[:2] =", (onT,knd,eps,noi),(cnt0,wrd),ebd[:2],pbd[:2], end = ", ")
            # - dump
            tal = "_".join([str(knd),str(eps),str(lbl)])
            mdl = "Noi_" + knd + "/"
            fle = Dir + mdl + "wrdKpbds_" + tal + ".pkl"
            if onT: fle += "_onT"
            with open(fle,"wb") as f:              pickle.dump(wrdKpbds,f)

    #:
    def m1_create_M(self, wrdKebds, epss, lbls, onT, typ=None):
        Dir = self._dir_DS_1 if typ == "DS" else (self._dir_OM_1 if typ == "OM" else None)
        knd = "M"
        # ----- pre
        with open(self._dir_common+"sigma_mat.pkl","rb") as f:          mat_sig = pickle.load(f)
        # ----- loop
        for eps,lbl in itertools.product(epss,lbls):
            # - set wrdKpbd, wrdKnbd9nrdsS   # pbd->perturbated ebd, nbd->near ebd, nrd->near wrd
            wrdKpbds = dict()
            for cnt0,(wrd,ebd) in enumerate(sorted(wrdKebds.items())):      #DD eps_t = eps if eps > 0 else 10**(-20)
                vec_noi = self._ut.Mahalanobis(eps,mat_sig)
                pbd     = ebd + vec_noi
                #DD if np.any(np.isnan(pbd)):       print("@@Bug?:",cnt0,wrd,ebd,pbd);  exit()
                # set
                wrdKpbds[wrd] = pbd
            # - dump
            tal = "_".join([str(knd),str(eps)]) if len(lbl) == 0 else "_".join([str(knd),str(eps),str(lbl)])
            mdl = "Noi_" + knd + "/"
            fle = Dir + mdl + "wrdKpbds_" + tal + ".pkl"
            if onT: fle += "_onT"
            with open(fle,"wb") as f:              pickle.dump(wrdKpbds,f)
    # ----- for main2_create_model.py
    def create_DS_NGJLM(self, knds, jacs, epss, lbls, onT):      # assumed that knds = ["AGv","AJ","M","AG","AL"]
        for lbl,knd,jac,eps in itertools.product(lbls,knds,jacs,epss):
            # ----- load
            tal = "_".join([str(knd),str(jac),str(eps),str(lbl)]) if knd in {"AGv","AG","AJ"} else "_".join([str(knd),str(eps),str(lbl)])
            mdl = "Noi_" + knd + "/"
            fle = self._dir_DS_1 + mdl + "wrdKpbds_" + tal + ".pkl"
            if onT: fle += "_onT"
            with open(fle,"rb") as f:              wrdKpbds = pickle.load(f)
            # ----- make lins
            lins = list()
            for wrd,pbd in sorted(wrdKpbds.items()):        #DD print("@@",wrd,pbd)
                lst = [wrd] + [ str(val) for val in pbd ]
                lin = " ".join(lst)
                lins.append(lin)
            # ----- write
            tal = "_".join([str(knd),str(jac),str(eps),str(lbl)])
            mdl = "Noi_" + knd + "/"
            fle = self._dir_DS_Mdl + mdl + "model_" + tal + ".txt"
            if onT: fle += "_onT"
            with open(fle, 'w') as f:
                for lin in lins:       f.write(lin + "\n")
    # ----- for main2_create_model.py
    def create_OM_NGJLM(self, knds, jacs, epss, lbls, onT):      # assumed that knds = ["AGv","AJ","M","AG","AL"]
        for lbl,knd,jac,eps in itertools.product(lbls,knds,jacs,epss):
            # ----- load
            tal = "_".join([str(knd),str(jac),str(eps),str(lbl)]) if knd in {"AGv","AG","AJ"} else "_".join([str(knd),str(eps),str(lbl)])
            mdl = "Noi_" + knd + "/"
            fle = self._dir_OM_1 + mdl + "wrdKpbds_" + tal + ".pkl"
            if onT: fle += "_onT"
            with open(fle,"rb") as f:              wrdKpbds = pickle.load(f)
            #:
            fle = self._dir_OM_9 + "cat9trmsS.pkl"
            with open(fle,"rb") as f:                 cat9trmsS = pickle.load(f)
            # ----- oddMan task
            cwaKfrqs = {"c":0,"w":0,"a":0}
            for cnt0,(cat,trms) in enumerate(cat9trmsS):
                x = self._ut.cwa_oddManTask(trms,wrdKpbds)   
                cwaKfrqs[x] += 1
            # ----- cal ratio
            ttl = sum(cwaKfrqs.values())
            cwaKrats = { x:float(frq)/ttl for x,frq in cwaKfrqs.items() }
            # ----- dump and write
            fle = self._dir_OM_CWA+mdl+"cwaKrats_"+tal+".pkl"
            if onT: fle += "_onT"
            with open(fle,"wb") as f:              pickle.dump(cwaKrats,f)
            #:
            lins = ["c "+str(cwaKrats["c"]),"w "+str(cwaKrats["w"]),"a "+str(cwaKrats["a"])]
            fle = self._dir_OM_CWA+mdl+"cwaKrats_"+tal+".txt"
            if onT: fle += "_onT"
            with open(fle, 'w') as f:
                for lin in lins:       f.write(lin + "\n")

    # ----- for mult process
    def mult_any(self, argsS = None, target = None, retType = None):   #### ? args_com = None
        tmeS = time.time()
        manager = Manager()
        Trds = len(argsS)
        # INFO
        print("[Mult], #threads =", Trds , end = ": ")
        # set rsltsS(St)
        if retType in {"list","lists"} :
            rsltsS = [manager.list() for i in range(Trds)]
        elif retType in {"dict","dicts"}:
            rsltsS = [manager.dict() for i in range(Trds)]
        else:
            print("[Mult] Error: retType is wrong;", retType);  return None
        # set argsS if None
        if argsS == None:
            print("[Mult] Error: argsS is wrong;", argsS);  return None
        # set jobs
        jobs = []
        cnt = manager.Value('i', 0)
        for i in range(Trds):       jobs.append(Process(target=target, args=(rsltsS[i], argsS[i], i+1, cnt)))
        # do jobs
        print("[Mult] now doing... ")
        for j in jobs:  j.start()
        for j in jobs:  j.join()
        # create retRslts
        if retType == "list":
            retRslts = list( itertools.chain.from_iterable( [rslts for rslts in rsltsS ] ) )
        elif retType == "dict":
            retRslts = dict( itertools.chain.from_iterable( [rslts.items() for rslts in rsltsS ] ) )
        elif retType == "lists":
            retRslts = [list(rslts) for rslts in rsltsS ]
        elif retType == "dicts":
            retRslts = [dict(rslts) for rslts in rsltsS ]
        # INFO
        ttl = int(time.time()-tmeS); h = ttl // int(3600); m = (ttl // int(60)) % int(60);  s = ttl % 60;
        #DD print("\n[Mult] summary: len(argsS),Trds,retType,len(retRslts) =", \
        #DD                          len(argsS),Trds,retType,len(retRslts), " (Total time:",h,"h",m,"m",s,"s)", end=" ")
        # return
        return retRslts

    # ----- for etc
    def pbd(self, ebd, knd, noi, rnd_stt=None):  # pbd->perturbated ebd
        # -- easy case
        if noi == 0:        return ebd
        # ------
        # -- init
        if rnd_stt == None:     rnd_stt = numpy.random      #.RandomState()     
        # -- cal nbd_r
        if   knd == "G":    # G->gauss
            nbd_r = ebd + rnd_stt.normal( loc=0, scale=noi, size=self._dimGlv)     ####numpy.random.normal(loc=0, scale=noi, size=self._dimGlv)  
        elif knd == "L":    # L->laplace
            nbd_r = ebd + rnd_stt.laplace(loc=0, scale=noi, size=self._dimGlv)     ####numpy.random.laplace(loc=0, scale=noi, size=self._dimGlv) 
        ## elif knd == "s":    # s->sign
        ##     inds_t  = set(rnd_stt.choice(range(self._dimGlv),size=noi,replace=False))
        ##     nbd_r = numpy.array([ val if ind not in inds_t else -val   for ind,val in enumerate(ebd)])
        ## elif knd == "z":    # z->zero
        ##     inds_t  = set(rnd_stt.choice(range(self._dimGlv),size=noi,replace=False))
        ##     nbd_r = numpy.array([ val if ind not in inds_t else 0.0   for ind,val in enumerate(ebd)])
        ##     #DD print(sorted(inds_t), [ (ind,ebd[ind],nbd_r[ind])  for ind in range(len(ebd)) if ebd[ind] != nbd_r[ind] ])
        else:
            print("ERROR: knd is wrong!; knd =", knd);      exit()
        # -- return
        return nbd_r
