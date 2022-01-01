#!/bin/bash

for dataset in aav_3 aav_4 aav_5 
do 
    for model in esm1b esm1v esm_rand 
    do
        python train_all.py $dataset $model 1 --mean --ensemble
        python train_all.py $dataset $model 1 --mut_mean --ensemble
        python train_all.py $dataset $model 1 --ensemble
    done
    python train_all.py $dataset cnn 1 --ensemble
done 