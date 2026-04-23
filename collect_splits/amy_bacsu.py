import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

n_muts = []
df = pd.read_csv('/home/kevyan/data/flip_data_zs/AMY_BACSU/splits/easy_split.csv', index_col=0)
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
    if row['n_mutations'] > 1:
        df.loc[i, 'set'] = "test"
    else:
        df.loc[i, 'set'] = "train"
        if np.random.random() < 0.15:
            df.loc[i, "validation"] = True
df.to_csv('/home/kevyan/data/flip_data_zs/AMY_BACSU/splits/one_to_many.csv', index=False)


df[df['set'] == "test"].shape
df['validation'].sum()
df[(df['set'] == 'train') & (~df['validation'])].shape
df[df['set'] == "test"]['target'].max()

df[df['validation']]['target'].max()
df[(df['set'] == 'train') & (~df['validation'])]['target'].max()
df['target'].max()
df[df['target'] > 0.19]
spearmanr(df['esm2_650M_scores'], df['dayhoff_3bur90_fwd'])