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
    # -- jaccard preserving noise func.
  
#    def jac_pre(self, voc, jac, top_k, vocs_sims) :
        #parameters
#        l_1   = 1.739094
#        l_2   = 2.00672
#        l_3   = 2.495598
#        l_4   = 3.347704
#        l_5   = 3.6020506
#        l_6   = 3.3456182
#        l_7   = 3.1546295
#        l_8   = 2.9447505
#        #calc sd
#        sim = numpy.average(numpy.array(vocs_sims[voc])[1:top_k])
#        if sim <= 0.3:
#            sd = -pow(sim,l_8/2 )*math.log(jac)
#        elif 0.3 < sim <= 0.35:
#            sd = -pow(sim,l_7/2 )*math.log(jac)
#        elif 0.35 < sim <= 0.4:
#            sd = -pow(sim,l_6/2 )*math.log(jac)
#        elif 0.4 < sim <= 0.45:
#            sd = -pow(sim,l_5/2 )*math.log(jac)
#        elif 0.45 < sim <= 0.5:
#            sd = -pow(sim,l_4/2 )*math.log(jac)
#        elif 0.5 < sim <= 0.55:
#            sd = -pow(sim,l_3/2 )*math.log(jac)
#        elif 0.55 < sim <= 0.6:
#            sd = -pow(sim,l_2/2 )*math.log(jac)
#        else:
#            sd = -pow(sim,l_1/2 )*math.log(jac)
#        return sd
    
    ####################
    # -- jaccard preserving noise func. 2-part ver.(sim)
#    def jac_pre_sim(self, voc, jac, top_k, vocs_sims) :
#        #parameters
#        l_1   = 2.68966224
#        l_2   = 3.325961223  
#        #calc sd  
#        sim = numpy.average(numpy.array(vocs_sims[voc])[1:top_k])        
#        if sim < 0.45:
#            sd = -pow(sim,l_2/2 )*math.log(jac)        
#        else:
#            sd = -pow(sim,l_1/2 )*math.log(jac)
#        return sd         
#    
    ####################
    # -- jaccard preserving noise func. 2-part ver.(dist)
#    def jac_pre_dist(self, voc, jac, top_k, vocs_norms) :
#        #parameters
#        l_1   = 1.8356207354  
#        l_2   = 1.2764672125
#        #calc sd
#        dist = numpy.average(numpy.array(vocs_norms[voc])[1:top_k])
#        if dist < 6:
#            sd = -pow(dist,-l_1/2 )*math.log(jac)
#        else:
#            sd = -pow(dist,-l_2/2 )*math.log(jac)
#        return sd 
    
#    def jac_pre_dist_var(self, voc, jac, top_k, vocs_norms, scale) :
#        #parameters
#        l_1   = 1.8356207354
#        l_2   = 1.2764672125
#        #calc sd
#        jac_mod = scale*jac
#        dist = numpy.average(numpy.array(vocs_norms[voc])[1:top_k])     # added by machide, because dist is not defined. Do check with Otake-san
#        if dist < 6:
#            sd = -pow(dist,-l_1/2 )*math.log(jac_mod)
#        else:
#            sd = -pow(dist,-l_2/2 )*math.log(jac_mod)
#        return sd

#    def jac_pre_dist_old1(self, voc, jac, top_k, vocs_norms) :
#        #parameters
#        l_1   = 2.3811115954
#        l_2   = 1.0883837886
#        #calc sd
#        dist = numpy.average(numpy.array(vocs_norms[voc])[1:top_k])
#        if dist < 6:
#            sd = -pow(dist,-l_1/2 )*math.log(jac)
#        else:
#            sd = -pow(dist,-l_2/2 )*math.log(jac)
#        return sd
   
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

#    def jac_pre_dist_analytic_gaussian_var(self, voc, para, alpha, voc_Delta, top_k, vocs_norms) :
#        #parameters
#        l_1   = 1.8356207354  
#        l_2   = 1.2764672125
#        #calc jac
#        sd_tent = alpha[para]*voc_Delta[voc]
#        dist = numpy.average(numpy.array(vocs_norms[voc])[1:top_k])
#        if dist < 6:
#            jac = math.exp(-pow(dist,l_1/2 )*sd_tent)
#        else:
#            jac = math.exp(-pow(dist,l_2/2 )*sd_tent)
#        #calc sd
#        if dist < 6:
#            sd = -pow(dist,-l_1/2 )*math.log(jac)
#        else:
#            sd = -pow(dist,-l_2/2 )*math.log(jac)
#        return sd 

#    #################### 
#    # -- unit ball noise func.
#
#    #calc norms
#    def calc_norm(self, vocs_ebd) :
#        vocs = list(vocs_ebd.keys())
#        norms_list = list()
#        for voc in vocs :
#            vec = vocs_ebd[voc]
#            norms = list()
#            for voc_oth in vocs :
#                vec_oth = vocs_ebd[voc_oth]
#                norm = numpy.linalg.norm(vec - vec_oth)
#                norms.append(norm)
#            norms_list.append(sorted(norms))
#        vocs_norms = dict(zip(vocs, norms_list))
#        return vocs_norms
#    
#    #divide vocs into two parts
#    def div_vocs_UB(self, vocs_norms, top_k = 10) :
#        vocs_1, vocs_2 = list(), list()
#        vocs = list(vocs_norms.keys())
#        for voc in vocs :
#            norms = vocs_norms[voc]
#            norms_k_mean = numpy.average(numpy.array(norms)[1:top_k])
#            if norms_k_mean < 6 :
#                vocs_1.append(voc)
#            else:
#                vocs_2.append(voc)
#        cats_vocs = {"1" : vocs_1, "2" : vocs_2}
#        return cats_vocs
#    
#    #calc radius_mean for unit_ball_2, unit_ball_3  
#    def calc_radius_mean_1(self, num, vocs, vocs_norms) :
#        radii = list()
#        for voc in vocs :
#            radius = vocs_norms[voc][int(num)]
#            radii.append(radius)
#        radius_mean = numpy.average(numpy.array(radii))
#        return radius_mean
#
#    #calc radius_mean for unit_ball_4
#    def calc_radius_mean_2(self, vocs_norms, nums = range(0,21)) :
#        vocs = list(vocs_norms.keys())
#        radius_mean = list()
#        for num in nums :
#            radius_num = list()
#            for voc in vocs :
#                radius = vocs_norms[voc][int(num)]
#                radius_num.append(radius)
#            radius_num_mean = numpy.average(numpy.array(radius_num))
#            radius_mean.append(radius_num_mean)
#        nums_radius_mean = dict(zip(nums, radius_mean))
#        return nums_radius_mean
#
#    #calc radius_mean for unit_ball_5 
#    def calc_radius_mean_3(self, vocs_norms, cats_vocs = None, top_k = 10, nums = range(0,21)) :
#        ut = UT()
#        if cats_vocs :
#            cats_vocs = cats_vocs
#        else :
#            cats_vocs = ut.div_vocs_UB(vocs_norms, top_k)
#        cats = list(cats_vocs.keys())
#        nums_radius_mean_list = list()
#        for cat in cats :
#            vocs = cats_vocs[cat]
#            radius_mean = list()
#            for num in nums :
#                radius_num = list()
#                for voc in vocs :
#                    radius = vocs_norms[voc][int(num)]
#                    radius_num.append(radius)
#                radius_num_mean = numpy.average(numpy.array(radius_num))
#                radius_mean.append(radius_num_mean)
#            nums_radius_mean = dict(zip(nums, radius_mean))
#            nums_radius_mean_list.append(nums_radius_mean)
#        cat_nums_radius_mean = dict(zip(cats, nums_radius_mean_list))
#        return cat_nums_radius_mean
#    
#    #calc sd for Danushka-san ver.
#    def unit_ball_1(self, voc, theta, vocs_norms, top_k = 10) :
#        vocs = list(vocs_norms.keys())
#        radius = 5.945585071378859
#        norms = vocs_norms[voc]
#        count = len([norm for norm in norms if norm <= theta * radius])
#        sd = 1/count
#        return sd
#
#    def unit_ball_1_var(self, voc, theta, vocs_norms, top_k=10, scale=None) :      # replace "scale" with "scale=None", because of SyntaxError: non-default argument follows default argument
#        vocs = list(vocs_norms.keys())
#        radius = 5.945585071378859
#        norms = vocs_norms[voc]
#        count = len([norm for norm in norms if norm <= theta * radius])
#        sd = pow(1/count, scale)
#        return sd
#    
#    #calc sd for Kawarabayashi-sensei ver_1.
#    def unit_ball_2(self, voc, num, vocs_norms, top_k = 10) :
#        ut = UT()
#        vocs = list(vocs_norms.keys())
#        radius = ut.calc_radius_mean_1(num, vocs, vocs_norms)
#        norms = vocs_norms[voc]
#        count = len([norm for norm in norms if norm <= radius])
#        sd = 1/count
#        return sd
#
#    #calc sd for Kawarabayashi-sensei ver_2.
#    def unit_ball_3(self, voc, num, vocs_norms, cats_vocs = None, top_k = 10) :
#        ut = UT()
#        if cats_vocs :
#            cats_vocs = cats_vocs
#        else : 
#            cats_vocs = ut.div_vocs_UB(vocs_norms, top_k)
#        vocs_1, vocs_2 = cats_vocs["1"], cats_vocs["2"]
#        if voc in vocs_1 :
#            vocs = vocs_1
#            radius = ut.calc_radius_mean_1(num, vocs, vocs_norms)
#        else :
#            vocs = vocs_2
#            radius = ut.calc_radius_mean_1(num, vocs, vocs_norms)
#        norms = vocs_norms[voc]
#        count = len([norm for norm in norms if norm <= radius])
#        sd = 1/count
#        return sd
#
#    #calc sd for Kawarabayashi-sensei ver_1.(all)
#    def unit_ball_4(self, voc, num, vocs_norms, nums_radius_mean = None, nums = range(0,21)) :
#        ut = UT()
#        if nums_radius_mean :
#            nums_radius_mean = nums_radius_mean
#        else : 
#            nums_radius_mean = ut.calc_radius_mean_2(vocs_norms, nums)
#        norms = vocs_norms[voc]
#        radius = nums_radius_mean[num]
#        count = len([norm for norm in norms if norm <= radius])
#        sd = 1/count
#        return sd
#    
#    #calc sd for Kawarabayashi-sensei ver_2.(all)
#    def unit_ball_5(self, voc, num, vocs_norms, cats_nums_radius_mean = None, cats_vocs = None, top_k = 10, nums = range(0,21)) :
#        ut = UT()
#        if cats_vocs :
#            cats_vocs = cats_vocs
#        else :
#            cats_vocs = ut.div_vocs_UB(vocs_norms, top_k)
#        if cats_nums_radius_mean :
#            cats_nums_radius_mean = cats_nums_radius_mean
#        else :
#            cats_nums_radius_mean = ut.calc_radius_mean_3(vocs_norms, cats_vocs, top_k, nums)
#        norms = vocs_norms[voc]
#        if voc in cats_vocs["1"] :
#            radius = cats_nums_radius_mean["1"][num]
#        else :
#            radius = cats_nums_radius_mean["2"][num]
#        count = len([norm for norm in norms if norm <= radius])
#        sd = 1/count
#        return sd           
#
#    def unit_ball_analytic_gaussian(self, voc, para, alpha, knd, vocs_norms) : #knd = 0.10 or 0.15 or 0.20 or 0.30 or 0.40 or 0.50
#        #parameters
#        radius_mean = 5.945585071378859
#        #calc count_vir
#        if knd == 0.10:
#            sd_tent = alpha[para]*10.088664096210367
#        if knd == 0.15:
#            sd_tent = alpha[para]*9.602291597840644
#        if knd == 0.20:
#            sd_tent = alpha[para]*9.585588802114408
#        if knd == 0.30:
#            sd_tent = alpha[para]*9.303671968257973
#        if knd == 0.40:
#            sd_tent = alpha[para]*8.882147129847692
#        if knd == 0.50:
#            sd_tent = alpha[para]*7.995965407151529
#        count_vir = math.floor(50/sd_tent)
#        #calc theta
#        vocs = list(vocs_norms.keys())
#        norms = vocs_norms[voc]
#        radius_voc = norms[count_vir-1]
#        theta = radius_voc / radius_mean
#        #calc sd
#        count = len([norm for norm in norms if norm <= theta * radius_mean])
#        sd = 50/count
#        return sd
#
#    def unit_ball_analytic_gaussian_var(self, voc, para, alpha, voc_Delta, vocs_norms) :
#        #parameters
#        radius_mean = 5.945585071378859
#        sd_tent = alpha[para]*voc_Delta[voc]
#        if sd_tent == 0:
#            sd = 0
#        else:
#            count_vir = math.floor(50/sd_tent)
#            #calc theta
#            vocs = list(vocs_norms.keys())
#            norms = vocs_norms[voc]
#            radius_voc = norms[count_vir-1]
#            theta = radius_voc / radius_mean
#            #calc sd
#            count = len([norm for norm in norms if norm <= theta * radius_mean])
#            sd = 50/count
#        return sd

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

#    # -- analytic_gaussian_noise_old_1
#    def analytic_gaussian_old_1(self, para, alpha):
#        sd = alpha[para]*11.2138118732622/math.sqrt(2*para)
#        return sd
#
#    # -- analytic_gaussian_noise_var_old_1
#    def analytic_gaussian_var_old_1(self, voc, para, alpha, voc_Delta):
#        sd = alpha[para]*voc_Delta[voc]/math.sqrt(2*para)
#        return sd        

