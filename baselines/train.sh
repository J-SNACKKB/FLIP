#!/bin/bash
python train_all.py gb1_4 esm1b 3 --mean
python train_all.py gb1_4 esm1v 3 --mean
python train_all.py gb1_4 esm_rand 3 --mean

python train_all.py gb1_4 esm1b 3
python train_all.py gb1_4 esm1v 3
python train_all.py gb1_4 esm_rand 3 
