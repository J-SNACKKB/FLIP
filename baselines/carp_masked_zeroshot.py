import argparse
import os
from typing import Tuple
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from sequence_models.pretrained import load_model_and_alphabet
from sequence_models.constants import MASK




def train(args: argparse.Namespace) -> None:

    # get the config, tokenizer, and model
    torch.cuda.set_device(args.gpu)
    DEVICE = torch.device('cuda:%d' % args.gpu)
    output_dir = os.path.join(args.out_fpath, args.landscape, "splits")
    carp, collator = load_model_and_alphabet('carp_640M')
    # Move only model to GPU
    model = carp.to(DEVICE)
    model = model.eval()
    a_to_t = collator.tokenizer.a_to_t
    seq_to_result = {}
    cache_file = os.path.join(output_dir, "carp640m_masked_scores.pt")
    if os.path.exists(cache_file):
        seq_to_result = torch.load(cache_file, weights_only=False)


    # Get files
    ## Grab data
    batch_size = 1
    landscape_path = os.path.join(args.data_fpath, args.landscape, "splits")
    split_csvs = os.listdir(landscape_path)
    split_csvs = [csv for csv in split_csvs if ".csv" in csv]
    print(split_csvs)


    for split_csv in split_csvs:
        df = pd.read_csv(os.path.join(landscape_path, split_csv))
        likelihoods = np.empty(len(df))
        # get the WT
        # wt_seq = df[df['variant_info'].isna()]['sequence'].values[0]
        for row in tqdm(df.itertuples(), total=len(df)):
            sequence = row.sequence
            variant_info = row.variant_info
            mut_sequence = sequence[:]
            input_sequence = sequence[:]
            if isinstance(variant_info, str):
                if "," in variant_info:
                    variant_info = variant_info.split(",")
                else:
                    variant_info = [variant_info]
                positions = tuple(int(v[1:-1]) - 1 for v in variant_info)
                old = tuple(v[0] for v in variant_info)
                new = tuple(v[-1] for v in variant_info)
            else:
                likelihoods[row.Index] = 0.0
                continue
            for i, pos in enumerate(positions):
                mut_sequence = mut_sequence[:pos] + new[i] + mut_sequence[pos + 1:]
                input_sequence = input_sequence[:pos] + MASK + input_sequence[pos + 1:]
            assert mut_sequence == sequence
            positions_key = ','.join(str(pos) for pos in positions)
            if positions_key not in seq_to_result:
                src = collator([[input_sequence]])[0].to(DEVICE)
                with torch.no_grad():
                    logits = model(src, repr_layers=[], logits=True)['logits'] # 1, ell, 30
                    logits = logits[0, positions] # n_positions, 30
                    log_probs = F.log_softmax(logits, dim=-1).cpu().detach().numpy()
                seq_to_result[positions_key] = log_probs
            log_probs = seq_to_result[positions_key]
            score = 0
            for i, lp in enumerate(log_probs):
                wt_lp = lp[a_to_t[old[i]]]
                mut_lp = lp[a_to_t[new[i]]]
                score += mut_lp - wt_lp
            likelihoods[row.Index] = score




        torch.save(seq_to_result, cache_file)
        if args.landscape == "PDZ3":
            df['carp_640m_zs_1'] = likelihoods[0]
            df['carp_640m_zs_2'] = likelihoods[1]
        else:
            df['carp_640m_masked_zs'] = likelihoods
        df.to_csv(os.path.join(output_dir, split_csv), index=False)



        if args.landscape != "PDZ3":
            print(split_csv,
                  spearmanr(df[df['set'] == 'test']['target'], df[df['set'] == 'test']['carp_640m_masked_zs']).statistic)
        # else:
        #     print(split_csv,
        #           spearmanr(df[df['set'] == 'test']['target'], df[df['set'] == 'test']['carp_640m_zs_1'] + df[df['set'] == 'test']['carp_640m_zs_2']).statistic)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_fpath', type=str)
    parser.add_argument('out_fpath', type=str)
    parser.add_argument('landscape', type=str)
    parser.add_argument('--gpu', type=int, default=1)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()





