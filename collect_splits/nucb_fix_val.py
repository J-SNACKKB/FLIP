import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
df = pd.read_csv('/home/kevyan/data/flip_data_20250814/flip_data/datasets/NucB/splits/medium.csv', index_col=0)
n_muts = []
for i, row in df.iterrows():
    muts = row['variant_info']
    if isinstance(muts, str):
        n_muts.append(len(row['variant_info'].split(',')))
    else:
        n_muts.append(0)
df['n_mutations'] = n_muts
np.random.seed(0)
df['validation'] = False
for i, row in df.iterrows():
    if row['n_mutations'] > 2:
        row['set'] = 'test'
    else:
        row['set'] = 'train'
        if np.random.random() < 0.15:
            df.loc[i, "validation"] = True

df.to_csv('/home/kevyan/data/flip_data_pruned/NucB/splits/two_to_many.csv', index=False)

df = pd.read_csv('/home/kevyan/data/flip_data_pruned/RhoMax/splits/by_wt.csv', index_col=0)
spearmanr(df['target'], df['esm2_650M_scores'])
spearmanr(df['target'], df['dayhoff_fwd'] + df['dayhoff_bwd'])
spearmanr(df['target'], df['dayhoff_3bur90_fwd'] + df['dayhoff_3bur90_bwd'])
spearmanr(df['target'], df['dayhoff_3bur90_fwd'] + df['dayhoff_3bur90_bwd'] + df['dayhoff_fwd'] + df['dayhoff_bwd'])