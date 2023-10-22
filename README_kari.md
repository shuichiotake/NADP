# Binary MZS calculator
 
# Introduction

This package is used for the experiments in the preprint (https://arxiv.org/abs/2205.13751), and targets the formal multiple zeta space spanned by formal symbols Z(a,b,c,...).
The symbols are binary (or mod 2) analogs to the shuffle-type regularized multiple zeta values (MZVs).
We call it the binary multiple zeta symbol or the binary MZS.
More precisely, for a weight,
- This makes extended double shuffle relations of shuffle type for the original multiple zeta values, whose coefficients are induced by modulo 2;
- This calculates generators and dimensions of the formal space and the graded formal spaces filtered by depth.

Note that we use the descending order for the definition as follows, where zeta(a,b,c,...) stands for original zeta values.
- zeta(2,1)=zeta(3).
- zeta(1,2) is divergent before reglarization.
 
# Requirement
 
* Python 3.9
* Cython 0.29.24 (which requires glibc)
* An unix shell (to use unix commands in processes)
 
Only environment under Anaconda3 (VER:2021.11,PLAT:linux-64) and CentOS Linux 7 is tested.
 
# Note

We use frozenset type for expressing linear relations because of modulo 2, i.e.,"+" and "-" are same, and even integer coefficients equal zero.
For example,
- The original relation "zeta(3)-zeta(2,1)=0" becomes "Z(3)+Z(2,1)=0" in modulo 2, which can be expressed as "frozenset({Z(3),Z(2,1)})".
- The original relation "zeta(4)-4zeta(3,1)=0" becomes "Z(4)=0" in modulo 2, which can be expressed as "frozenset({Z(4)})".
 
If we use verbose mode, the calculating time and memory usage are stocked in "../Data/Wkk/mdlKhmss.txt" with a module-key, where "kk" means a weight.
 
# Author
 
* Tomoya Machide
 
# License
 
MIT license
 
Enjoy the bynary formal multiple zeta space. Thank you!
