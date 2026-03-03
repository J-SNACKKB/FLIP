import os
from collections import Counter

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

flip_path = '/home/kevyan/results/flipv3/'
# results_path = os.path.join(flip_path, 'combined_flip_results')
# result_files = os.listdir(results_path)





# Make clean full landscape zero shot score csvs
pruned_path = "/home/kevyan/data/flip_data_pruned/"
landscapes = os.listdir(pruned_path)
os.makedirs(os.path.join(flip_path, "zs"), exist_ok=True)
for landscape in landscapes:
    if landscape == "rhomax":
        df = pd.read_csv(os.path.join(pruned_path, landscape, "splits", "by_wt.csv"))
    elif landscape == "TrpB":
        df = pd.read_csv(os.path.join(pruned_path, landscape, "splits", "by_position.csv"))
    else:
        df = pd.read_csv(os.path.join(pruned_path, landscape, "splits", "random.csv"))
    columns = [c for c in df.columns if c not in ("set", "validation", "dayhoff_min", "dayhoff_3bur90_min")]
    df = df[columns]
    if landscape != "PDZ3":
        df.rename(columns={"dayhoff_fwd": "dayhoff_3bgrhmc_fwd", "dayhoff_bwd": "dayhoff_3bgrhmc_bwd"}, inplace=True)
        df['dayhoff'] = (df['dayhoff_3bgrhmc_fwd'] + df['dayhoff_3bgrhmc_bwd'] + df['dayhoff_3bur90_fwd'] + df['dayhoff_3bur90_bwd']) / 4
    else:
        df.rename(columns={"dayhoff_fwd_1": "dayhoff_3bgrhmc_fwd_1", "dayhoff_bwd_1": "dayhoff_3bgrhmc_bwd_1"}, inplace=True)
        df.rename(columns={"dayhoff_fwd_2": "dayhoff_3bgrhmc_fwd_2", "dayhoff_bwd_2": "dayhoff_3bgrhmc_bwd_2"}, inplace=True)
        d = (df['dayhoff_3bgrhmc_fwd_1'] + df['dayhoff_3bgrhmc_bwd_1'] + df['dayhoff_3bur90_fwd_1'] + df['dayhoff_3bur90_bwd_1']) / 4
        d += (df['dayhoff_3bgrhmc_fwd_2'] + df['dayhoff_3bgrhmc_bwd_2'] + df['dayhoff_3bur90_fwd_2'] + df['dayhoff_3bur90_bwd_2']) / 4
        d /= 2
        df['dayhoff'] = d
        df['esm2_650M_scores'] = (df['esm2_650M_scores1'] + df['esm2_650M_scores2'].fillna(0)) / 2
        df['carp_640m_zs'] = (df['carp_640m_zs_1'] + df['carp_640m_zs_2']) / 2
    if 'carp_640m_masked_zs' in df.columns:
        df['carp_640m_zs'] = df['carp_640m_masked_zs']
        df = df.drop(columns=['carp_640m_masked_zs'])
    rho1 = spearmanr(df['target'], df['esm2_650M_scores']).statistic
    rho2 = spearmanr(df['target'], df['dayhoff']).statistic
    rho3 = spearmanr(df['target'], df['carp_640m_zs']).statistic

    print(landscape, rho1, rho2, rho3)
    df.to_csv(os.path.join(flip_path, "zs", landscape + "_zs.csv"), index=False)

# Make clean prediction csvs for each test set
prediction_path = os.path.join(flip_path, "all_predictions")
for landscape in landscapes:
    os.makedirs(os.path.join(prediction_path, landscape), exist_ok=True)
    split_csvs = os.listdir(os.path.join(pruned_path, landscape, "splits"))
    split_csvs = [c for c in split_csvs if ".csv" in c]
    for split_csv in split_csvs:
        df = pd.read_csv(os.path.join(pruned_path, landscape, "splits", split_csv), index_col=0)
        df = df[df['set'] == 'test']
        columns = [c for c in df.columns if c not in ("set", "validation", "dayhoff_min", "dayhoff_3bur90_min", "Unnamed: 0")]
        df = df[columns]
        if landscape != "PDZ3":
            df.rename(columns={"dayhoff_fwd": "dayhoff_3bgrhmc_fwd", "dayhoff_bwd": "dayhoff_3bgrhmc_bwd"},
                      inplace=True)
            df['dayhoff'] = (df['dayhoff_3bgrhmc_fwd'] + df['dayhoff_3bgrhmc_bwd'] + df['dayhoff_3bur90_fwd'] + df[
                'dayhoff_3bur90_bwd']) / 4
        else:
            df.rename(columns={"dayhoff_fwd_1": "dayhoff_3bgrhmc_fwd_1", "dayhoff_bwd_1": "dayhoff_3bgrhmc_bwd_1"},
                      inplace=True)
            df.rename(columns={"dayhoff_fwd_2": "dayhoff_3bgrhmc_fwd_2", "dayhoff_bwd_2": "dayhoff_3bgrhmc_bwd_2"},
                      inplace=True)
            d = (df['dayhoff_3bgrhmc_fwd_1'] + df['dayhoff_3bgrhmc_bwd_1'] + df['dayhoff_3bur90_fwd_1'] + df[
                'dayhoff_3bur90_bwd_1']) / 4
            d += (df['dayhoff_3bgrhmc_fwd_2'] + df['dayhoff_3bgrhmc_bwd_2'] + df['dayhoff_3bur90_fwd_2'] + df[
                'dayhoff_3bur90_bwd_2']) / 4
            d /= 2
            df['dayhoff'] = d
            df['esm2_650M_scores'] = (df['esm2_650M_scores1'] + df['esm2_650M_scores2'].fillna(0)) / 2
            df['carp_640m_zs'] = (df['carp_640m_zs_1'] + df['carp_640m_zs_2']) / 2
        if 'carp_640m_masked_zs' in df.columns:
            df['carp_640m_zs'] = df['carp_640m_masked_zs']
            df = df.drop(columns=['carp_640m_masked_zs'])
        plm_path = os.path.join(flip_path, "plm_predictions", landscape + "_" + split_csv[:-4] + "_predictions.csv")
        if os.path.isfile(plm_path):
            df2 = pd.read_csv(plm_path)
            if landscape == "PDZ3":
                df2['sequence'] = df['sequence'].values
            df3 = df.merge(df2, left_on='sequence', right_on='sequence')
            df4 = pd.read_csv(os.path.join(flip_path, "ridge", landscape, split_csv))
            df4.rename(columns={'prediction': 'Ridge', "target": "linear_target"}, inplace=True)
            df3 = df3.merge(df4, left_on='sequence', right_on='sequence')
            df4 = pd.read_csv(os.path.join(flip_path, "ridge_zs", landscape, split_csv))
            df4 = df4[['sequence', 'prediction']]
            df4.rename(columns={'prediction': 'zsRidge'}, inplace=True)
            df3 = df3.merge(df4, left_on='sequence', right_on='sequence')
            df3.to_csv(os.path.join(prediction_path, landscape, split_csv), index=False)

# Make csv for metrics with all replicates
nice_models = {
    'Ridge': ("Ridge", False, 0),
    "zsRidge": ("zsRidge", False, 0),
    "dayhoff": ("Dayhoff", False, 0),
    "carp_640m_zs": ("CARP-640M zero shot", False, 0),
    "esm2_650M_scores": ("ESM2-650M", False, 0),
}
for seed in range(5):
    for model in ['carp', 'esmc']:
        for p in ['pretrained', 'naive']:
            key = '_'.join([model, p, str(seed)])
            if model == "carp":
                nice_models[key] = ("CARP-640M", p == "pretrained", seed)
            else:
                nice_models[key] = ("ESMC-300M", p == "pretrained", seed)

all_results = pd.DataFrame(columns=['dataset', "split", "model", "pretrained", "Spearman", "MSE", "NDCG", "seed"])
for landscape in landscapes:
    pred_csvs = os.listdir(os.path.join(prediction_path, landscape))
    for pred_csv in pred_csvs:
        pred_df = pd.read_csv(os.path.join(prediction_path, landscape, pred_csv))
        split = pred_csv[:-4]
        for key in nice_models:
            if key in pred_df.columns:
                idx = len(all_results)
                all_results.loc[idx, ['dataset', 'split']] = (landscape, split)
                all_results.loc[idx, ['model', 'pretrained', 'seed']] = nice_models[key]
                all_results.loc[idx, ["Spearman"]] = spearmanr(pred_df['scaled_target'], pred_df[key]).correlation
                if "carp" in key or "esmc" in key or "Ridge" in key:
                    all_results.loc[idx, ["MSE"]] = ((pred_df['scaled_target'] - pred_df[key]) ** 2).mean()
                pos_targets = (pred_df['target'] - pred_df['target'].min()).values[None, :]
                all_results.loc[idx, ["NDCG"]] = ndcg_score(pos_targets, pred_df[key].values[None, :])
all_results.to_csv(os.path.join(flip_path, "all_metrics.csv"), index=False)

# Make csv with replicates aggregated
grouped = all_results.groupby(['dataset', 'split', 'model', 'pretrained'])
agged = grouped.agg(
    spearman_mean=('Spearman', np.mean),
    spearman_std=('Spearman', np.std),
    mse_mean=('MSE', np.mean),
    mse_std=('MSE', np.std),
    ndcg_mean=('NDCG', np.mean),
    ndcg_std=('NDCG', np.std),
)


compiled = agged.reset_index()
print(compiled)
compiled.to_csv(os.path.join(flip_path, 'all_metrics_aggregated.csv'), index=True)

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
    "Ridge (one-hot)",
    'Ridge (one-hot + likelihoods)',
    "Dayhoff likelihood",
    "ESM2-650M likelihood",
    "CARP-640M likelihood",
    "CARP-640M supervised",
    "CARP-640M naive supervised",
    "ESMC-300M supervised",
    "ESMC-300M naive supervised",
]

for i, row in compiled.iterrows():
    if row['model'] in model_dict:
        compiled.loc[i, 'model'] = model_dict[row['model']]
    else:
        compiled.loc[i, 'model'] = model_dict[(row['model'], row['pretrained'])]

all_tasks = []
for i, row in compiled.iterrows():
    all_tasks.append((row['dataset'], row['split']))

all_tasks = set(all_tasks)
for task in all_tasks:
    if 'random' in task:
        continue
    print(task)
    comp = compiled[(compiled['dataset'] == task[0]) & (compiled['split'] == task[1])]
    for model in model_order:
        c = comp[comp['model'] == model]
        for i, row in c.iterrows():
            if 'supervised' in model:
                print(row['model'] + '& $%.3f \pm %.3f$ & $%.3f \pm %.3f$\\\\' %(row['spearman_mean'], row['spearman_std'], row['ndcg_mean'], row['ndcg_std']))

            else:
                print(row['model'] + '& $%.3f$ & $%.3f$\\\\' %(row['spearman_mean'], row['ndcg_mean']))
by_task = compiled.pivot(index=['dataset', 'split'], columns=['model', 'pretrained'])
by_task = by_task.reset_index()
by_task.head()
(by_task['spearman_mean']['CARP-640M'][True] > by_task['spearman_mean']['CARP-640M'][False]).sum()
(by_task['spearman_mean']['ESMC-35M'][True] > by_task['spearman_mean']['ESMC-35M'][False]).sum()
(by_task['spearman_mean']['Ridge'][False] > by_task['spearman_mean']['zsRidge'][False]).sum()
(by_task['spearman_mean']['Dayhoff'][False] > by_task['spearman_mean']['zsRidge'][False]).sum()
(by_task['spearman_mean']['ESM2-650M'][False] > by_task['spearman_mean']['zsRidge'][False]).sum()

model_list = by_task.columns[2:11]
best_spearmans = []
for i, row in by_task.iterrows():
    print(row['dataset'].values[0], row['split'].values[0], *model_list[row['spearman_mean'].argmax()][1:])
    best_spearmans.append(model_list[row['spearman_mean'].argmax()][1:])
Counter(best_spearmans)
#
# best_mses = []
# for i, row in by_task.iterrows():
#     print(row['dataset'].values[0], row['split'].values[0], *model_list[row['mse_mean'].argmin()][1:])
#     best_mses.append(model_list[row['mse_mean'].argmin()][1:])
# Counter(best_mses)
#
#
# by_task[['dataset', 'split', 'spearman_mean']]
# by_task['spearman_mean'].mean()
# print("dataset,split,p>n,p>r")
# for dataset in set(compiled.dataset):
#     for split in set(compiled[compiled['dataset'] == dataset].split):
#         r = compiled[(compiled['dataset'] == dataset) & (compiled['split'] == split) & (compiled['model'] == 'Ridge')]['spearman_mean'].values[0]
#         p = compiled[(compiled['dataset'] == dataset) & (compiled['split'] == split) & (compiled['model'] == 'CARP-640M') & (compiled['pretrained'])]['spearman_mean'].values[0]
#         n = compiled[(compiled['dataset'] == dataset) & (compiled['split'] == split) & (compiled['model'] == 'CARP-640M') & (~compiled['pretrained'])]['spearman_mean']
#         if len(n) > 0:
#             n = n.values[0]
#         else:
#             n = -np.inf
#         print(dataset + "," + split + "," + str(p > n) + "," + str(p > r))