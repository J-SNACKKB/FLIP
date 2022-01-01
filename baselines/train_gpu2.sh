for dataset in meltome_mixed meltome_human meltome_humancell 
do 
    for model in esm1b esm1v esm_rand 
    do
        python train_all.py $dataset $model 2 --mean --ensemble
        python train_all.py $dataset $model 2 --ensemble
    done
    python train_all.py $dataset cnn 2 --ensemble
done 

