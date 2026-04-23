import os
from pathlib import Path
import subprocess
import argparse
import os

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("split_path", type=str, help="Directory to download raw data")
parser.add_argument("out_path", type=str, help="Directory to save processed file")
args = parser.parse_args()

split_path = Path(args.split_path)
# split_path = "/home/kevyan/src/FLIPv3/splits/rhomax"
os.makedirs(split_path, exist_ok=True)
subprocess.call(["wget", "-P", split_path, "https://github.com/dina-lab3D/OpsiGen/raw/refs/heads/colab/excel/data.xlsx"])

df = pd.read_excel(os.path.join(split_path, "data.xlsx"))

grouped = df.groupby("Wildtype")
grouped = grouped.agg({"lmax": ['mean', 'count']})
grouped.columns = grouped.columns.to_flat_index()
grouped = grouped.sort_values(('lmax', 'count'))
grouped = grouped.reset_index()
val_wt = []
n_val = 0
test_wt = []
n_test = 0
num_val = 100
num_test = 175
current = 'val'

for wildtype in grouped['Wildtype'].values:
    if current == 'val' and n_val < num_val:
        val_wt.append(wildtype)
        n_val += grouped[grouped['Wildtype'] == wildtype][('lmax', 'count')].values[0]
        if n_test < num_test:
            current = 'test'
    elif current == 'test' and n_test < num_test:
        test_wt.append(wildtype)
        n_test += grouped[grouped['Wildtype'] == wildtype][('lmax', 'count')].values[0]
        if n_val < num_val:
            current = 'val'

df_val = df[df['Wildtype'].isin(val_wt)]
df_test = df[df['Wildtype'].isin(test_wt)]
df_train = df[~df['Wildtype'].isin(np.concatenate([val_wt, test_wt]))]
print(len(df_val), "validation samples", "mean = ", df_val['lmax'].mean())
print(len(df_test), "Test samples", "mean = ", df_test['lmax'].mean())
print(len(df_train), "Train samples", "mean = ", df_train['lmax'].mean())

df_out = pd.DataFrame()
df_out['wildtype'] = df['Wildtype']
df_out['sequence'] = df['Sequence']
df_out['target'] = df['lmax']
df_out['set'] = 'train'
df_out.loc[df_out['wildtype'].isin(test_wt), 'set'] = 'test'
df_out['validation'] = df_out['wildtype'].isin(val_wt)
df_out = df_out[['sequence', 'target', 'set', 'validation']]
df_out.to_csv(os.path.join(os.path.join(args.out_path, "by_wt.csv")), index=False)