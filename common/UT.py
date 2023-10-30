#from multiprocessing import Process
import numpy,itertools,functools,random,time,math,subprocess,pickle
from statistics import mean, median, variance, stdev

################################################################
class UTRoot:
    # -- initialize
    def __init__(self):
        pass

################################################################
class UT(UTRoot):
    # initialize
    def __init__(self, onTMode=False):
        UTRoot.__init__(self)
        # --

    ####################
    # -- test function
    def test_print(self, stg="test in UT"):     print(stg)
    
    ####################
   
    def jac_pre_dist_analytic_gaussian(self, voc, para, alpha, knd, top_k, vocs_norms) : #knd = 0.10 or 0.15 or 0.20 or 0.30 or 0.40 or 0.50
        #parameters
        l_1   = 1.8356207354  
        l_2   = 1.2764672125
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
        dist = numpy.average(numpy.array(vocs_norms[voc])[1:top_k])
        if dist < 6:
            jac = math.exp(-pow(dist,l_1/2 )*sd_tent)
        else:
            jac = math.exp(-pow(dist,l_2/2 )*sd_tent)
        #calc sd
        if dist < 6:
            sd = -pow(dist,-l_1/2 )*math.log(jac)
        else:
            sd = -pow(dist,-l_2/2 )*math.log(jac)
        return sd

    ####################
    #calc skewness
    def skewness(self, jacs_list) :
        m = numpy.mean(jacs_list)
        n = len(jacs_list)
        s = numpy.std(jacs_list)
        if s >0:
            X = [pow((x - m)/s, 3) for x in jacs_list]
            skewness = n*numpy.sum(numpy.array(X))/((n-1)*(n-2))
        else:
            skewness = 0
        return skewness

    ####################
    # -- for odd man task
    """  argument1: 5 terms such as [term1,term2,term3,term4,term5]
                      * term is a sequence of vocablaries with blank or _ separators; e.g., whirlpool, data_sets
                      * term1 should be correct answer
         argument2: dictonary(key=vocablary,value=embedding vector)
         #--
         return:    characters "c", "w" or "a"
                      * "c" means this solver answer is term1, i.e., correct.
                      * "w" means this solver answer is not term1, i.e., wrong.
                      * "a" means some vocablaries appearing in argument1 are not covered by argument2; this solver does not work, and abstained. 
                        "a" is also returned if the arguments are missed.
　　"""
    def cwa_oddManTask(self, trms, vocKebds):    # c->correct, w->wrong, a->abstained        # Suppose that len(trms) == 5
        ctrm       = trms[0]  # ctrm->correct answer trm
        trmKvocs4S = { trm:[ voc for voc in trm.replace(" ","_").split("_")]  for trm in trms }      # note; type of vocs4 is list
        # -----
        # - check whether trms are covered by vocKpbds
        fset  = { voc in vocKebds for vocs4 in trmKvocs4S.values() for voc in vocs4 }
        vocsL = [ voc for vocs4 in trmKvocs4S.values() for voc in vocs4 ];      vocs = set(vocsL)
        #: 
        if False in fset:       return "a"      # return "a" because not covered; 
        if len(trms) != 5:      return "a"      #            because #trms !=5
        if len(vocsL) != 5:     return "a"      #            because there is a term consisting of 2 more vocablaries
        if "running." in vocs:  return "a"      #            because a term includes "."
        # -----
        # - cal aves of cos-sims of trms\{trm} for trm in trms
        trmKcohs = dict()       # coh->cohesion in SS5.1 of the paper "Spot the Odd Man Out"
        for trm_r in trms:      # r->root
            trms_e = [ trm for trm in trms if trm != trm_r ]     # e->excluded
            trm1,trm2,trm3,trm4 = trms_e
            # cal each val
            vals = list()   #DD print("@@0",trm_r,trms_e)
            for voc1,voc2,voc3,voc4 in itertools.product(trmKvocs4S[trm1],trmKvocs4S[trm2],trmKvocs4S[trm3],trmKvocs4S[trm4]):
                lst = [voc1,voc2,voc3,voc4] #DD print("@@1",lst)
                vals.append( sum([ self.cos(vocKebds[vocA],vocKebds[vocB]) for vocA,vocB in itertools.combinations(lst,2) ]) )
            # set max val 
            trmKcohs[trm_r] = max(vals)
        # - get predicted term
        ptrm = sorted(trmKcohs.keys(), key=lambda trm:trmKcohs[trm] )[-1]       #DD print("@@@@", trms,ctrm,trmKcohs,ptrm,ctrm == ptrm)
        # - return "C" or "W"; C->Correct, W->Wrong
        return "c" if ctrm == ptrm else "w"
    def cos(self, vec1, vec2, val_err=None):
        nrm1,nrm2 = numpy.linalg.norm(vec1),numpy.linalg.norm(vec2)
        inr       = numpy.dot(vec1,vec2)
        # - cal
        val = inr/(nrm1*nrm2) if nrm1*nrm2 > 0 else val_err
        # - return
        return val          #HH if val <= 1.0 else 1.0

    ####################
    # -- Mahalanobis noise
    def Mahalanobis(self, para, sigma_mat):
        X = numpy.random.multivariate_normal(numpy.zeros(300),numpy.identity(300))
        N = numpy.reshape(X/numpy.linalg.norm(X),(300,1))
        Y = numpy.random.gamma(300,1/para)
        Z = Y*numpy.dot(sigma_mat,N)
        noise_vec = numpy.ravel(Z)
        return noise_vec
    
    ####################
    # -- analytic_gaussian_noise
    def analytic_gaussian(self, para, alpha, knd): #knd = 0.10 or 0.15 or 0.20 or 0.30 or 0.40 or 0.50
        if knd == 0.10:
            sd = alpha[para]*10.088664096210367
        if knd == 0.15:
            sd = alpha[para]*9.602291597840644
        if knd == 0.20:
            sd = alpha[para]*9.585588802114408
        if knd == 0.30:
            sd = alpha[para]*9.303671968257973
        if knd == 0.40:
            sd = alpha[para]*8.882147129847692
        if knd == 0.50:
            sd = alpha[para]*7.995965407151529
        return sd

    # -- analytic_gaussian_noise_var
    def analytic_gaussian_var(self, voc, para, alpha, voc_Delta): 
        sd = alpha[para]*voc_Delta[voc]
        return sd
    
    # -- analytic_laplace_noise 
    def analytic_laplace(self, para):
        sc = 151.7991981/para       # sc->scale,Delta=151.7991981,para=ε 
        return sc        

