#!/bin/bash

for dataset in aav_5 aav_7 meltome_mixed meltome_human meltome_humancell 
do
    python train_all.py $dataset cnn 3 --ensemble
done 