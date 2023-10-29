# NADP 
 
# Introduction

This package makes experimental results in the paper "A Neighbourhood-Aware Differential Privacy Mechanism for Static Word Embeddings".
 
# Requirement
 
* Python 3.9
 
Only environment under Anaconda3 (VER:2021.11,PLAT:linux-64) is tested.
 
# Experiments

1. To make the data set in common

   The initial given data are vocKebds_benchmark_TW_73404.pkl and vocs_ebd_dict_10593.pkl. To make another data set in common, please execute the python file neighbourhoods.py in the directory noise_gen.
   The pickle data, alpha_mod.pkl, sigma_mat.pkl, voc_Delta_0.10.pkl, voc_Delta_0.20.pkl, voc_Delta_0.30.pkl, voc_Delta_0.40.pkl, voc_Delta_0.50.pkl will be created in common.  
  
2. To output the graph of skewness

   We can create the skewness data of the Jaccard index between the top_10 nearest neibhgour of a word and that of a perturbed word.
   Please execute the python file skewness.py in the directory noise_gen.
   The pickle data skewness.pkl and its graph skewness.png will be created in the directory /noise_gen/data/.  

3. To make data for downstream tasks
   We should excute the two files in /downStream_oddMan/downStream/Main_DS/:
   - main0_init.py
   - main1_create.py
     
The first file is for the pickle file "wrdKnrms4S.pkl" in ~~/Data_DS/0_Init, where ~~ means /downStream_oddMan/downStream/.
For each word, the file gives the norms from the orher words by the ascending order. The second file creates the embedding data for using 'SentEval' (see https://github.com/facebookresearch/SentEval) in ~~/Data_DS/Models. The created file of the data consists of 73404 lines, and the line has one word at the top and 300 real numbers after that, separeted by the bland. Temporal data for embeddings with noise are created in ~~/Data_DS/1_VocKpebd. 

Note that every file is the test mode in default, and change "onT = True" to "onT = False" in the beginning part of the file for the actual thing.
 
4. To make data for the odd man task
 
# Authors
 
* Danushka Bollegala, Shuichi Otake, Tomoya Machide, Ken-ichi Kawarabayashi
 
# License
 
Apache License 2.0
