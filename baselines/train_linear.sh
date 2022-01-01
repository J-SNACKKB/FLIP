#!/bin/bash
for dataset in aav_1 aav_2 aav_3 aav_4 aav_5 aav_5 aav_6 aav_7 meltome_mixed meltome_human meltome_humancell meltome_humancell gb1_1 gb1_2 gb1_3 gb1_4 gb1_5 
do 
    python train_all.py $dataset ridge --gb1_shorten
done 

