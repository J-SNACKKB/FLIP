#!/bin/bash

#python train_linear.py aav_dt1 3 
#python train_linear.py aav_dt2 3
#python train_linear.py aav_nt1 3
#python train_linear.py aav_nt2 3 

python train_esm.py meltome_clust esm1b_rand 3 --mean
python train_esm.py meltome_full esm1b_rand 3 --mean
python train_esm.py meltome_mixed esm1b_rand 3 --mean
python train_esm.py aav_dt1 esm1b_rand 3 --mean 
python train_esm.py aav_nt1 esm1b_rand 3 --mean 
python train_esm.py aav_nt1 esm1b_rand 3 --mean

