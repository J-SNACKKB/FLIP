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

from sequence_models.collaters import Seq2PropertyCollater
from sequence_models.constants import PAD, START, STOP
from sequence_models.structure import Attention1d
from sequence_models.utils import warmup
from sequence_models.flip_utils import load_flip_data
from sequence_models.pretrained import load_model_and_alphabet




def train(args: argparse.Namespace) -> None:

    # get the config, tokenizer, and model
    torch.cuda.set_device(args.gpu)
    DEVICE = torch.device('cuda:%d' % args.gpu)
    output_dir = os.path.join(args.out_fpath, args.landscape, "splits")
    carp, collator = load_model_and_alphabet('carp_640M')
    # Move only model to GPU
    model = carp.to(DEVICE)
    model = model.eval()
    seq_to_result = {}
    cache_file = os.path.join(output_dir, "carp640m_scores.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            seq_to_result = json.load(f)

    # Get files
    ## Grab data
    batch_size = 1
    landscape_path = os.path.join(args.data_fpath, args.landscape, "splits")
    split_csvs = os.listdir(landscape_path)
    split_csvs = [csv for csv in split_csvs if ".csv" in csv]
    print(split_csvs)


    for split_csv in split_csvs:
        df = pd.read_csv(os.path.join(landscape_path, split_csv))
        likelihoods = [np.empty(len(df)), np.empty(len(df))]
        for row in tqdm(df.itertuples(), total=len(df)):
            sequence = row.sequence
            if ":" in sequence:
                sequences = sequence.split(":")
            else:
                sequences = [sequence]
            for j, sequence in enumerate(sequences):
                if sequence not in seq_to_result:
                    if len(sequence) == 0:
                        seq_to_result[sequence] = 0
                        continue
                    src = collator([[sequence]])[0].to(DEVICE)
                    with torch.no_grad():
                        logits = model(src, repr_layers=[], logits=True)['logits']
                        out = F.cross_entropy(logits[0], src.flatten())
                        seq_to_result[sequence] = -out.detach().cpu().item()
                    likelihoods[j][row.Index] = seq_to_result[sequence]
                else:
                    likelihoods[j][row.Index] = seq_to_result[sequence]
        with open(cache_file, "w") as f:
            json.dump(seq_to_result, f)
        if args.landscape == "PDZ3":
            df['carp_640m_zs_1'] = likelihoods[0]
            df['carp_640m_zs_2'] = likelihoods[1]
        else:
            df['carp_640m_zs'] = likelihoods[0]
        df.to_csv(os.path.join(output_dir, split_csv), index=False)



        if args.landscape != "PDZ3":
            print(split_csv,
                  spearmanr(df[df['set'] == 'test']['target'], df[df['set'] == 'test']['carp_640m_zs']).statistic)
        else:
            print(split_csv,
                  spearmanr(df[df['set'] == 'test']['target'], df[df['set'] == 'test']['carp_640m_zs_1'] + df[df['set'] == 'test']['carp_640m_zs_2']).statistic)




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





