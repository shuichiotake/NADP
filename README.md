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
   
   We should excute the two files in ~~/Main_DS/:
   - main0_init.py;
   - main1_create.py.
Here ~~ means /downStream_oddMan/downStream/.
     
The first file is for the pickle file "wrdKnrms4S.pkl" in ~~/Data_DS/0_Init.
For each word, the file gives the norms from the orher words by the ascending order. 

The second file creates text files of embedding data in ~~/Data_DS/Models, which are used in the libraly 'SentEval' producing downstream tasks (https://github.com/facebookresearch/SentEval). The created text file consists of 73404 lines, and each line has one word at the top and 300 real numbers after that, separeted by the blank. Temporal data for embeddings with noise are also created in ~~/Data_DS/1_VocKpebd. 

Note that the two executable files are on the test mode in default, and change "onT = True" to "onT = False" in the beginning part of the files for the actual thing.
 
4. To make data for the odd man task

   We should excute the two files in ~~/Main_OM/:
   - main0_init.py
   - main1_create.py
Here ~~ means /downStream_oddMan/oddMan/.
  
The first file is for the pickle file "wrdKnrms4S.pkl" in ~~/Data_OM/0_Init, as in making data of downstream tasks.
For each word, the file gives the norms from the orher words by the ascending order. 

The second file creates results of odd man tasks in ~~/Data_OM/CWAs; for this, we use data set in the paper "Spot the Odd Man Out: Exploring the Associative Power of Lexical Resources" by Gabriel Stanovsky and Mark Hopkins (https://github.com/gabrielStanovsky/odd-man-out) with some modifications for our experiments, which are stored in ~~/Data_OM/9_FromPaper. The result consists of 3 lines, each line has a character in {'c','w','a'} at the top and its ratio separeted by the blank: 'c' means correct, 'w' means wrong, 'a' means abstained. Temporal data for embeddings with noise are also created in ~~/Data_DS/1_VocKpebd. 

Note that the two executable files are on the test mode in default, and change "onT = True" to "onT = False" in the beginning part of the files for the actual thing.
 
# Authors
 
* Danushka Bollegala, Shuichi Otake, Tomoya Machide, Ken-ichi Kawarabayashi
 
# License
 
Apache License 2.0
