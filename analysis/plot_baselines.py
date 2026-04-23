import os
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
_ = sns.set(font_scale=1.7)
_ = sns.set_style('white')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
flip_path = '/home/kevyan/results/flipv3/'
pruned_path = "/home/kevyan/data/flip_data_pruned/"

datasets = os.listdir(os.path.join(flip_path, "all_predictions"))


df = pd.read_csv(os.path.join(flip_path, "random_ridge.csv"))

df_zs = pd.DataFrame()
zs_model_dict = {
    'dayhoff': 'Dayhoff',
    'carp_640m_zs': 'CARP-640M',
    # 'carp_640m_masked_zs': 'CARP-640M',
    'esm2_650M_scores': 'ESM2-650M'
}
idx = 0
for dataset in datasets:
    predictions = pd.read_csv(os.path.join(flip_path, 'zs', dataset + '_zs.csv'))
    for model in zs_model_dict.keys():
        if model in predictions.columns:
            df_zs.loc[idx, 'dataset'] = dataset
            df_zs.loc[idx, 'model'] = zs_model_dict[model]
            df_zs.loc[idx, 'Spearman'] = spearmanr(predictions['target'], predictions[model]).correlation
            idx += 1
best_zs = df_zs.groupby('dataset').agg({'Spearman': 'max'}).reset_index()

# Get number of training examples in each split
dataset_path = '/home/kevyan/data/flip_data_pruned/'
df_sizes = pd.DataFrame(columns=['dataset', 'split', 'n_train', 'n_valid', 'n_test'])
for dataset in datasets:
    split_csvs = os.listdir(os.path.join(dataset_path, dataset, 'splits'))
    split_csvs = [c for c in split_csvs if c[-4:] == '.csv']
    for split_csv in split_csvs:
        df_data = pd.read_csv(os.path.join(dataset_path, dataset, 'splits', split_csv))
        n_test = len(df_data[df_data['set'] == 'test'])
        n_train = len(df_data[(df_data['set'] == 'train') & (~df_data['validation'])])
        n_valid = len(df_data[df_data['validation']])
        df_sizes.loc[len(df_sizes)] = [dataset, split_csv[:-4], n_train, n_valid, n_test]
print(df_sizes)

df_metrics = pd.read_csv(os.path.join(flip_path, 'all_metrics.csv'))
model_dict = {
    "Ridge": "Ridge (one-hot)",
    "zsRidge": 'Ridge (one-hot + likelihoods)',
    "Dayhoff": "Dayhoff likelihood",
    "ESM2-650M": "ESM2-650M likelihood",
    "CARP-640M zero shot": "CARP-640M likelihood",
    ("CARP-640M", True): "CARP-640M supervised",
    ("CARP-640M", False): "CARP-640M naive supervised",
    ("ESMC-300M", True): "ESMC-300M supervised",
    ("ESMC-300M", False): "ESMC-300M naive supervised",
}

for i, row in df_metrics.iterrows():
    if row['model'] in model_dict:
        df_metrics.loc[i, 'model'] = model_dict[row['model']]
    else:
        df_metrics.loc[i, 'model'] = model_dict[(row['model'], row['pretrained'])]
df_metrics = df_metrics.fillna(0)
split_dict = {
    'close_to_far': 'position',
    'far_to_close': 'position',
    'by_mutation': 'mutation',
    'by_position': 'position',
    'by_wt': 'wild type',
    'random': 'random',
    'one_to_many': 'number',
    'to_P06241': 'wild type',
    'to_P01053': 'wild type',
    'to_P0A9X9': 'wild type',
    'low_to_high': 'fitness',
    'three_to_many': 'number',
    'single_to_double': 'number',
    'two_to_many': 'number',
}
pal = [
    '#76B900',
    '#A77BB5',
    '#4E79A7',
    '#FF8A80',
    '#F28E2B',
    '#E15759'
]
split_hues = {"number": pal[0], "wild type": pal[1], "position": pal[2], "mutation": pal[3], "fitness": pal[4]}


for i, dataset in enumerate(datasets):
    fig, ax = plt.subplots()
    _ = sns.lineplot(data=df[df['dataset'] == dataset], x='n_train', y='Spearman', color='grey', style='model',
                     markers=True, ax=ax, ms=20, alpha=0.8)
    _ = ax.axhline(y=best_zs[best_zs['dataset'] == dataset]['Spearman'].values[0], color='grey', linestyle='-',
                   label='best zero-shot likelihood score')
    _ = ax.semilogx()
    _ = ax.set_xlabel('Number of training examples')
    _ = ax.set_ylim([-0.35, 1])
    split_csvs = os.listdir(os.path.join(dataset_path, dataset, 'splits'))
    split_csvs = [c for c in split_csvs if c[-4:] == '.csv']
    for split_csv in split_csvs:
        if 'random' in split_csv:
            continue
        split_name = split_csv[:-4]
        if dataset == 'Amylase' and split_name == 'by_position':
            split_name = 'by_mutation'
        color = split_hues[split_dict[split_name]]
        x = df_sizes[(df_sizes['dataset'] == dataset) & (df_sizes['split'] == split_csv[:-4])]['n_train']
        y1 = df_metrics[(df_metrics['dataset'] == dataset) & (df_metrics['split'] == split_csv[:-4]) & (df_metrics['model'] == 'Ridge (one-hot)')]['Spearman'].values[0]
        _ = ax.plot(x, y1, 'o', color=color, ms=20, alpha=0.7, mew=0)
        y2 = df_metrics[(df_metrics['dataset'] == dataset) & (df_metrics['split'] == split_csv[:-4]) & (df_metrics['model'] == 'Ridge (one-hot + likelihoods)')]['Spearman'].values[0]
        _ = ax.plot(x, y2, 'x', color=color, ms=20, alpha=1.0, mew=4)
    _ = ax.set_title(dataset)
    legend = ax.legend(title='Model')
    legend.remove()
    fig.savefig(os.path.join(flip_path, "plots", "random_ridge_%s.pdf" %dataset), dpi=300, bbox_inches='tight')
handles, labels = ax.get_legend_handles_labels()
fig, ax = plt.subplots()
legend = ax.legend(handles, labels, title='Model')
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(os.path.join(flip_path, 'plots', 'random_ridge_models.pdf'), dpi=300, bbox_inches=bbox)
elements = [
    Patch(facecolor='gray', edgecolor=None, label='random'),
]
elements += [Patch(facecolor=split_hues[s], edgecolor=None, label=s, alpha=0.7) for s in split_hues]
legend = ax.legend(handles=elements, title='Split type')
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(os.path.join(flip_path, 'plots', 'random_ridge_split_types.pdf'), dpi=300, bbox_inches=bbox)
