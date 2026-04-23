import os
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

pal = sns.color_palette()
models = [
    ('dayhoff', 'Dayhoff likelihood', pal[0]),
    ('esm2_650M_scores', "ESM2-650M likelihood", pal[1]),
    ('carp_640m_zs', "CARP-640M likelihood", pal[2]),
    ('Ridge', 'Ridge (one-hot)', pal[3]),
    ('zsRidge', 'Ridge (one-hot + likelihoods)', pal[4]),
    ('carp_naive', 'CARP-640M naive supervised', pal[5]),
    ('carp_pretrained', 'CARP-640M supervised', pal[6]),
    ('esmc_naive', 'ESMC-300M naive supervised', pal[7]),
    ('esmc_pretrained', 'ESMC-300M supervised', pal[8]),
]
os.makedirs(os.path.join(flip_path, 'plots', 'predictions'), exist_ok=True)
# plot individual predictions
_ = sns.set(font_scale=1)
_ = sns.set_style('white')
for dataset in datasets:
    splits = os.listdir(os.path.join(flip_path, "all_predictions", dataset))
    for split in splits:
        split_name = split[:-4]
        df = pd.read_csv(os.path.join(flip_path, 'all_predictions', dataset, split))
        df['carp_naive'] = (df['carp_naive_0'] + df['carp_naive_1'] + df['carp_naive_2'] + df['carp_naive_3'] + df['carp_naive_4']) / 5
        df['carp_pretrained'] = (df['carp_pretrained_0'] + df['carp_pretrained_1'] + df['carp_pretrained_2'] + df['carp_pretrained_3'] + df['carp_pretrained_4']) / 5
        df['esmc_naive'] = (df['esmc_naive_0'] + df['esmc_naive_1'] + df['esmc_naive_2'] + df['esmc_naive_3'] + df['esmc_naive_4']) / 5
        df['esmc_pretrained'] = (df['esmc_pretrained_0'] + df['esmc_pretrained_1'] + df['esmc_pretrained_2'] + df['esmc_pretrained_3'] + df['esmc_pretrained_4']) / 5
        for ugly, pretty, color in models:
            # if dataset == "PDZ3":
            #     if ugly in ['dayhoff', 'esm2_650M_scores', 'carp_640m_zs']:
            #         continue
            fig, ax = plt.subplots()
            _ = sns.scatterplot(data=df, x='scaled_target', y=ugly, alpha=0.3, marker='o', ax=ax, color='gray')
            _ = ax.set_ylabel(pretty)
            _ = ax.set_xlabel('scaled target')
            fig.savefig(os.path.join(flip_path, 'plots', 'predictions',
                                     '_'.join([dataset, split_name, ugly + '.png'])),
                        dpi=100, bbox_inches='tight')

# Plot landscape zero-shot scores
_ = sns.set(font_scale=1.7)
_ = sns.set_style('white')
pal = [
    '#76B900',
    '#A77BB5',
    '#4E79A7',
    '#FF8A80',
    '#F28E2B',
    '#E15759'
]
model_dict = {
    'dayhoff': 'Dayhoff likelihood',
    'carp_640m_zs': 'CARP-640M likelihood',
    'esm2_650M_scores': 'ESM2-650M likelihood',
}
model_order = [
    'Dayhoff',
    'ESM2-650M',
    'CARP-640M',
]
plot_me = pd.DataFrame()
idx = 0
for dataset in datasets:
    predictions = pd.read_csv(os.path.join(flip_path, 'zs', dataset + '_zs.csv'))
    for model in model_dict.keys():
        if model in predictions.columns:
            plot_me.loc[idx, 'dataset'] = dataset
            plot_me.loc[idx, 'split'] = 'full'
            plot_me.loc[idx, 'model'] = model_dict[model]
            plot_me.loc[idx, 'Spearman'] = spearmanr(predictions['target'], predictions[model]).correlation
            idx += 1
dataset_order = ['Amylase', 'IRED', 'NucB', 'TrpB', 'Hydro', 'Rhomax', 'PDZ3']
fig, ax = plt.subplots(figsize=(8, 6))
palette = ['gray', 'gray', 'gray']
_ = sns.barplot(x='dataset', y='Spearman', data=plot_me, ax=ax, hue='model', palette=palette, order=dataset_order)
hatches = ['/', '.', '*']
handles, labels = ax.get_legend_handles_labels()
for i, bar in enumerate(ax.patches[:21]):
    hatch = hatches[i // 7]
    bar.set_hatch(hatch)
for i, handle in enumerate(handles):
    handle.set_hatch(hatches[i])
ax.legend(handles, labels)
ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
fig.savefig(os.path.join(flip_path, 'plots', 'dataset_zeroshot.pdf'), dpi=300, bbox_inches='tight')


# plot aggregate things
df = pd.read_csv(os.path.join(flip_path, 'all_metrics.csv'))
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
model_order = [
    "Dayhoff likelihood",
    "ESM2-650M likelihood",
    "CARP-640M likelihood",
    "Ridge (one-hot)",
    "Ridge (one-hot + likelihoods)",
    "CARP-640M naive supervised",
    "CARP-640M supervised",
    "ESMC-300M naive supervised",
    "ESMC-300M supervised",
]
# hue = {m: p for m, p in zip(model_order, pal)}


for i, row in df.iterrows():
    if row['model'] in model_dict:
        df.loc[i, 'model'] = model_dict[row['model']]
    else:
        df.loc[i, 'model'] = model_dict[(row['model'], row['pretrained'])]

# for dataset in set(df['dataset']):
#     for split in set(df[df['dataset'] == dataset]['split']):
#         pretty_split = split.replace('_', '-')
#         data = df[(df['dataset'] == dataset) & (df['split'] == split)]
#         fig, ax = plt.subplots()
#         _ = sns.barplot(data=data, x='model', y='Spearman', hue='model', hue_order=model_order, palette=hue,
#                         errorbar=None, ax=ax, alpha=0.7, order=model_order)
#         _ = sns.stripplot(data=data, x='model', y='Spearman', hue='model', hue_order=model_order, palette=hue,
#                           ax=ax, order=model_order)
#         _ = ax.set_title(dataset + " " + pretty_split)
#         _ = ax.set_ylim([-0.2, 1])
#         ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
#         fig.savefig(os.path.join(flip_path, 'plots', 'spearman_' + dataset + '_' + split + '.pdf'),
#                     dpi=300, bbox_inches='tight')


split_dict = {
    'close_to_far': 'position',
    'far_to_close': 'position',
    'by_position': 'position',
    'by_mutation': 'mutation',
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
split_hues = {"full": 'gray', "number": pal[0], "wild type": pal[1], "position": pal[2], "mutation": pal[3], "fitness": pal[4]}
split_order = ["full", 'number', 'wild type', 'position', 'mutation', 'fitness']
df['split type'] = df.apply(lambda row: split_dict[row['split']], axis=1)

all_zs = df[df['model'].isin(['Dayhoff likelihood', 'ESM2-650M likelihood', 'CARP-640M likelihood'])]
all_zs = all_zs[['dataset', 'split', 'model', 'Spearman']]
all_zs = pd.concat([plot_me, all_zs], ignore_index=True)

for idx, row in all_zs.iterrows():
    all_zs.loc[idx, 'task'] = row['dataset'] + ' ' + '-'.join(row['split'].split('_'))
    all_zs.loc[idx, 'split type'] = split_dict[row['split']] if row['split'] in split_dict else row['split']

task_order = [
    'Amylase full',
    'Amylase one-to-many',
    'Amylase close-to-far',
    'Amylase far-to-close',
    'Amylase by-mutation',
    'IRED full',
    'IRED two-to-many',
    'NucB full',
    'NucB two-to-many',
    'TrpB full',
    'TrpB one-to-many',
    'TrpB two-to-many',
    'TrpB by-position',
    'Hydro full',
    'Hydro three-to-many',
    'Hydro low-to-high',
    'Hydro to-P06241',
    'Hydro to-P0A9X9',
    'Hydro to-P01053',
    'Rhomax full',
    'Rhomax by-wt',
    'PDZ3 full',
    'PDZ3 single-to-double'
]
all_zs['task'] = pd.Categorical(all_zs['task'], categories=task_order, ordered=True)
_ = sns.set(font_scale=1.5)
_ = sns.set_style('white')
fig, ax = plt.subplots(figsize=(16, 6))
# _ = ax.fill_betweenx([-0.5, 0.67], x1=[-1.1, -1.1], x2=[4.5, 4.5], color='gray', alpha=0.1)
_ = ax.fill_betweenx([-0.5, 0.67], x1=[4.5, 4.5], x2=[6.5, 6.5], color='gray', alpha=0.1, linewidth=0)
_ = ax.fill_betweenx([-0.5, 0.67], x1=[8.5, 8.5], x2=[12.5, 12.5], color='gray', alpha=0.1, linewidth=0)
_ = ax.fill_betweenx([-0.5, 0.67], x1=[18.5, 18.5], x2=[20.5, 20.5], color='gray', alpha=0.1, linewidth=0)

_ = sns.scatterplot(data=all_zs, x='task', y='Spearman', hue='split type', hue_order=split_order,
                   ax=ax, style='model', palette=split_hues, s=150, markers=['X', 'o', 's'], alpha=0.7)
_ = ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
_ = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
fig.savefig(os.path.join(flip_path, 'plots', 'all_zeroshot.pdf'), dpi=300, bbox_inches='tight')

_ = sns.set(font_scale=1.7)
_ = sns.set_style('white')
split_hues = {"number": pal[0], "wild type": pal[1], "position": pal[2], "mutation": pal[3], "fitness": pal[4]}
split_order = ['number', 'wild type', 'position', 'mutation', 'fitness']
plot_me = pd.DataFrame()
idx = 0
tasks = list(df[['dataset', 'split']].values)
tasks = set([(t[0], t[1]) for t in tasks])
for dataset, split in tasks:
    current = df[(df['dataset'] == dataset) & (df['split'] == split)]
    for j, row in current.iterrows():
        plot_me.loc[idx, 'dataset'] = dataset
        plot_me.loc[idx, 'split'] = split
        plot_me.loc[idx, row['model']] = row['Spearman']
        plot_me.loc[idx, 'split type'] = split_dict[split]
    idx += 1
plot_me = plot_me.fillna(0)
plot_me = plot_me[plot_me['split type'] != 'random']
models = list(plot_me.columns[2:])
models.remove('split type')
models = np.array(models)
plot_me['best model'] = models[plot_me[models].values.argmax(axis=1)]
print(plot_me[['dataset', 'split', 'best model']].sort_values('dataset'))
# Plot Ridge vs Ridge + zs
fig, ax = plt.subplots()
_ = ax.plot([-0.2, 0.8], [-0.2, 0.8], color='gray')
_ = sns.scatterplot(data=plot_me, x='Ridge (one-hot)', y='Ridge (one-hot + likelihoods)', hue='split type',
                    alpha=0.7, ax=ax, color='gray', hue_order=split_order, s=150, legend=False, palette=split_hues)
_ = ax.set_ylabel('Ridge (one-hot+likelihoods)')
fig.savefig(os.path.join(flip_path, 'plots', 'ridge_comparison.pdf'), dpi=300, bbox_inches='tight')

# Plot Ridge + zs vs best zs
plot_me['best zero-shot'] = plot_me.loc[:, ['Dayhoff likelihood', 'CARP-640M likelihood', 'ESM2-650M likelihood']].max(axis=1)
plot_me['best zs method'] = np.array(['Dayhoff likelihood', 'CARP-640M likelihood', 'ESM2-650M likelihood'])[plot_me.loc[:, ['Dayhoff likelihood', 'CARP-640M likelihood', 'ESM2-650M likelihood']].values.argmax(axis=1)]
fig, ax = plt.subplots()
_ = ax.plot([-0.2, 0.8], [-0.2, 0.8], color='gray')
_ = sns.scatterplot(data=plot_me, x='best zero-shot', y='Ridge (one-hot + likelihoods)', hue='split type',
                    alpha=0.7, ax=ax, palette=split_hues, hue_order=split_order, s=150)
_ = ax.set_ylabel('Ridge (one-hot+likelihoods)')

legend = ax.legend()
legend.remove()
fig.savefig(os.path.join(flip_path, 'plots', 'zs_vs_ridgezs.pdf'), dpi=300, bbox_inches='tight')

# Plot Ridge vs best PLM
plot_me['best PLM'] = plot_me.loc[:, ['CARP-640M supervised', 'CARP-640M naive supervised', 'ESMC-300M supervised', 'ESMC-300M naive supervised']].max(axis=1)
archs = np.array(['CARP-640M', 'CARP-640M', 'ESMC-300M', 'ESMC-300M'])
plot_me['best architecture'] = archs[plot_me.loc[:, ['CARP-640M supervised', 'CARP-640M naive supervised', 'ESMC-300M supervised', 'ESMC-300M naive supervised']].values.argmax(axis=1)]
fig, ax = plt.subplots()
_ = ax.plot([-0.2, 0.8], [-0.2, 0.8], color='gray')
_ = sns.scatterplot(data=plot_me, x='Ridge (one-hot)', y='best PLM', hue='split type', style='best architecture',
                    alpha=0.7,
                    legend=False, style_order=['CARP-640M', 'ESMC-300M'], ax=ax, palette=split_hues,
                    hue_order=split_order, s=150)
fig.savefig(os.path.join(flip_path, 'plots', 'ridge_vs_plm.pdf'), dpi=300, bbox_inches='tight')

fig, ax = plt.subplots()
_ = ax.plot([-0.2, 0.8], [-0.2, 0.8], color='gray')
_ = sns.scatterplot(data=plot_me, x='Ridge (one-hot + likelihoods)', y='best PLM', hue='split type',
                    style='best architecture', style_order=['CARP-640M', 'ESMC-300M'], legend=False,
                    alpha=0.7, ax=ax, palette=split_hues, hue_order=split_order, s=150)
fig.savefig(os.path.join(flip_path, 'plots', 'ridgezs_vs_plm.pdf'), dpi=300, bbox_inches='tight')

# Plot CARP
fig, ax = plt.subplots()
_ = ax.plot([-0.2, 0.8], [-0.2, 0.8], color='gray')
_ = sns.scatterplot(data=plot_me, x='CARP-640M likelihood', y='CARP-640M supervised', hue='split type',
                    alpha=0.7, ax=ax, palette=split_hues, hue_order=split_order, s=150)
legend = ax.legend()
legend.remove()
fig.savefig(os.path.join(flip_path, 'plots', 'carp.pdf'), dpi=300, bbox_inches='tight')
handles, labels = ax.get_legend_handles_labels()
fig, ax = plt.subplots()
legend = ax.legend(handles, labels, title='Split type')
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(os.path.join(flip_path, 'plots', 'split_types.pdf'), dpi=300, bbox_inches=bbox)

# Get dataset sizes
dataset_path = '/home/kevyan/data/flip_data_pruned/'
for dataset in set(plot_me['dataset']):
    split_csvs = os.listdir(os.path.join(dataset_path, dataset, 'splits'))
    split_csvs = [c for c in split_csvs if c[-4:] == '.csv']
    for split_csv in split_csvs:
        df_data = pd.read_csv(os.path.join(dataset_path, dataset, 'splits', split_csv))
        n_test = len(df_data[df_data['set'] == 'test'])
        n_train = len(df_data[(df_data['set'] == 'train') & (~df_data['validation'])])
        n_valid = len(df_data[df_data['validation']])
        index = plot_me[(plot_me['dataset'] == dataset) & (plot_me['split'] == split_csv[:-4])].index
        if not index.empty:
            plot_me.loc[index, 'n_train'] = n_train



# Plot pretrained vs naive
plot_me2 = pd.DataFrame()
idx = 0
for i, row in plot_me.iterrows():
    plot_me2.loc[idx, 'dataset'] = row['dataset']
    plot_me2.loc[idx, 'split'] = row['split']
    plot_me2.loc[idx, 'split type'] = split_dict[row['split']]
    plot_me2.loc[idx, 'pretrained'] = row['CARP-640M supervised']
    plot_me2.loc[idx, 'naive'] = row['CARP-640M naive supervised']
    plot_me2.loc[idx, 'architecture'] = 'CARP-640M'
    idx += 1
    plot_me2.loc[idx, 'dataset'] = row['dataset']
    plot_me2.loc[idx, 'split'] = row['split']
    plot_me2.loc[idx, 'split type'] = split_dict[row['split']]
    plot_me2.loc[idx, 'pretrained'] = row['ESMC-300M supervised']
    plot_me2.loc[idx, 'naive'] = row['ESMC-300M naive supervised']
    plot_me2.loc[idx, 'architecture'] = 'ESMC-300M'
    idx += 1
fig, ax = plt.subplots()
_ = ax.plot([-0.2, 0.8], [-0.2, 0.8], color='gray')
_ = sns.scatterplot(data=plot_me2, x='naive', y='pretrained', hue='split type', style='architecture', alpha=0.7,
                    palette=split_hues, ax=ax, style_order=['CARP-640M', 'ESMC-300M'], s=150, legend=True,
                    hue_order=split_order)
legend = ax.legend()
legend.remove()
handles, labels = ax.get_legend_handles_labels()
fig.savefig(os.path.join(flip_path, 'plots', 'plms.pdf'), dpi=300, bbox_inches='tight')
fig, ax = plt.subplots()
legend = ax.legend(handles, labels)
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(os.path.join(flip_path, 'plots', 'plms_legend.pdf'), dpi=300, bbox_inches=bbox)

# Barplot of ridge spearmans
fig, ax = plt.subplots(figsize=(8, 6))
plot_me3 = df[df['model'] == 'Ridge (one-hot)']
plot_me3 = df[df['split'] != 'random']
for i, row in plot_me3.iterrows():
    plot_me3.loc[i, 'task'] = row['dataset'] + ' ' + row['split'].replace('_', '-')
task_order = [
    'Amylase one-to-many',
    'Amylase close-to-far',
    'Amylase far-to-close',
    'Amylase by-mutation',
    'IRED two-to-many',
    'NucB two-to-many',
    'TrpB one-to-many',
    'TrpB two-to-many',
    'TrpB by-position',
    'Hydro three-to-many',
    'Hydro low-to-high',
    'Hydro to-P06241',
    'Hydro to-P0A9X9',
    'Hydro to-P01053',
    'Rhomax by-wt',
    'PDZ3 single-to-double'
]
_ = sns.barplot(data=plot_me3, x='task', y='Spearman', ax=ax, hue='split type', hue_order=split_order,
                palette=split_hues,
                order=task_order, legend=False)
_ = ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
fig.savefig(os.path.join(flip_path, 'plots', 'all_ridge.pdf'), dpi=300, bbox_inches='tight')


plot_me3 = df[df['model'].isin(['Dayhoff likelihood', 'ESM-650M likelihood', 'CARP-640M likelihood'])]
for i, row in plot_me3.iterrows():
    plot_me3.loc[i, 'task'] = row['dataset'] + ' ' + row['split'].replace('_', '-')
plot_me3 = plot_me3.groupby('task').agg({'Spearman': 'mean', 'split type': lambda x: x.values[0]})
plot_me3 = plot_me3.reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
_ = sns.barplot(data=plot_me3, x='task', y='Spearman', ax=ax, hue='split type', hue_order=split_order,
                palette=split_hues, legend=False, order=task_order)
_ = ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
fig.savefig(os.path.join(flip_path, 'plots', 'best_zs.pdf'), dpi=300, bbox_inches='tight')


# Plot train zs Spearman vs test zs Spearman
datasets = os.listdir(pruned_path)
model_order = [
    "Dayhoff",
    "ESM2-650M",
    "CARP-640M",
    "Ridge (one-hot)",
    "Ridge (one-hot + likelihoods)",
    "CARP-640M naive supervised",
    "CARP-640M supervised",
    "ESMC-300M naive supervised",
    "ESMC-300M supervised",
]
hue = {m: p for m, p in zip(model_order, pal)}

zs_df = pd.DataFrame()
zs_columns = {
    'esm2_650M_scores': "ESM2-650M",
    'carp_640m_zs': "CARP-640M",
    'dayhoff': 'Dayhoff',}
idx = 0
for dataset in datasets:
    for split in os.listdir(os.path.join(pruned_path, dataset, 'splits')):
        if split[-4:] != ".csv":
            continue
        results = pd.read_csv(os.path.join(pruned_path, dataset, 'splits', split))
        if 'carp_640m_masked_zs' in results.columns:
            results['carp_640m_zs'] = results['carp_640m_masked_zs']
            results = results.drop(columns=['carp_640m_masked_zs'])
        if dataset != "PDZ3":
            results.rename(columns={"dayhoff_fwd": "dayhoff_3bgrhmc_fwd", "dayhoff_bwd": "dayhoff_3bgrhmc_bwd"}, inplace=True)
            results['dayhoff'] = (results['dayhoff_3bgrhmc_fwd'] + results['dayhoff_3bgrhmc_bwd'] + results['dayhoff_3bur90_fwd'] + results['dayhoff_3bur90_bwd']) / 4
        else:
            results.rename(columns={"dayhoff_fwd_1": "dayhoff_3bgrhmc_fwd_1", "dayhoff_bwd_1": "dayhoff_3bgrhmc_bwd_1"}, inplace=True)
            results.rename(columns={"dayhoff_fwd_2": "dayhoff_3bgrhmc_fwd_2", "dayhoff_bwd_2": "dayhoff_3bgrhmc_bwd_2"}, inplace=True)
            d1 = (results['dayhoff_3bgrhmc_fwd_1'] + results['dayhoff_3bgrhmc_bwd_1'] + results['dayhoff_3bur90_fwd_1'] + results['dayhoff_3bur90_bwd_1']) / 4
            d1 += (results['dayhoff_3bgrhmc_fwd_2'] + results['dayhoff_3bgrhmc_bwd_2'] + results['dayhoff_3bur90_fwd_2'] + results['dayhoff_3bur90_bwd_2']) / 4
            d1 /= 2
            results['dayhoff'] = d1
            results['esm2_650M_scores'] = (results['esm2_650M_scores1'] + results['esm2_650M_scores2'].fillna(0)) / 2
            results['carp_640m_zs'] = (results['carp_640m_zs_1'] + results['carp_640m_zs_2']) / 2
        for m in zs_columns.keys():
            if m in results.columns:
                zs_df.loc[idx, 'dataset'] = dataset
                zs_df.loc[idx, 'split type'] = split_dict[split[:-4]]
                zs_df.loc[idx, 'split'] = split[:-4]
                zs_df.loc[idx, 'model'] = zs_columns[m]
                for s in ['train', 'test']:
                    d = results[results['set'] == s]
                    sp = spearmanr(d['target'], d[m]).correlation
                    zs_df.loc[idx, s + ' Spearman'] = sp
                idx += 1
plot_me = zs_df[zs_df['split'] != 'random']
fig, ax = plt.subplots()
_ = ax.plot([-0.2, 0.7], [-0.2, 0.7], color='gray')
_ = sns.scatterplot(data=plot_me, x='train Spearman', y='test Spearman', hue='split type',
                    palette=split_hues, hue_order=split_order, ax=ax, alpha=0.7, s=150, legend=False)
fig.savefig(os.path.join(flip_path, 'plots', 'zs_train_v_test.pdf'), dpi=300, bbox_inches='tight')
print("Train/test zero shot spearmans")
print(pearsonr(zs_df.dropna()['train Spearman'], zs_df.dropna()['test Spearman']))
zs_df = zs_df[zs_df['split type'] != 'random']
for model in model_order[:3]:
    d = zs_df[zs_df['model'] == model]
    print(model, pearsonr(d.dropna()['train Spearman'], d.dropna()['test Spearman']))

for st in set(split_dict.values()):
    if st != 'random':
        d = zs_df[zs_df['split type'] == st]
        print(st, pearsonr(d.dropna()['train Spearman'], d.dropna()['test Spearman']).correlation)
