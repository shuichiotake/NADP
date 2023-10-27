# NADP 
 
# Introduction

This package makes experimental results in the paper "A Neighbourhood-Aware Differential Privacy Mechanism for Static Word Embeddings".
 
# Requirement
 
* Python 3.9
 
Only environment under Anaconda3 (VER:2021.11,PLAT:linux-64) is tested.
 
# Experiments

1. To make the data set in common
   To make the data set in common, please execute the python file neighbourhoods.py in the directory noise_gen.
   The pickle data, alpha_mod.pkl, sigma_mat.pkl, voc_Delta_0.10.pkl, voc_Delta_0.20.pkl, voc_Delta_0.30.pkl, voc_Delta_0.40.pkl, voc_Delta_0.50.pkl will be created in common.  
  
2. To output the graph of skewness
   We can create the skewness data of the Jaccard index between the top_10 nearest neibhgour of a word and that of a perturbed word.
   Please excute the python file skewness.py in the directory noise_gen.
   The pickle data skewness.pkl and its graph skewness.png will be created in the directory /noise_gen/data/.  

3. To make data for downstream tasks
4. To make data for the odd man task
 
# Authors
 
* Danushka Bollegala, Shuichi Otake, Tomoya Machide, Ken-ichi Kawarabayashi
 
# License
 
Apache License 2.0
