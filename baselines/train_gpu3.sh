#!/bin/bash

for dataset in gb1_1 gb1_2 gb1_3 gb1_4 gb1_5 aav_6 aav_7
do 
    for model in esm1b esm1v esm_rand
    do
        python train_all.py $dataset $model 3 --mean --ensemble
        python train_all.py $dataset $model 3 --mut_mean --gb1_shorten --ensemble
        python train_all.py $dataset $model 3 --gb1_shorten --ensemble
    done
    python train_all.py $dataset cnn 3 --gb1_shorten --ensemble
done p