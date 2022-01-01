#!/bin/bash

for dataset in aav_3 aav_4 aav_5 aav_6 aav_7 gb1_1 gb1_2 gb1_3 gb1_4 gb1_5 
do 
    python train_all.py $dataset cnn 0 --gb1_shorten
    for model in esm1b esm1v esm_rand
    do
        python train_all.py $dataset $model 0 --mean 
        python train_all.py $dataset $model 0 --mut_mean --gb1_shorten
        python train_all.py $dataset $model 0 --gb1_shorten
    done
done 

for dataset in meltome_mixed meltome_human meltome_humancell 
do 
    python train_all.py $dataset cnn 0 
    for model in esm1b esm1v esm_rand
    do
        python train_all.py $dataset $model 0 --mean
        python train_all.py $dataset $model 0 
    done
done 

