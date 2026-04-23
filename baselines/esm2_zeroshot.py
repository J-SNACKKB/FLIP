import argparse
import os
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import esm

def is_amlt() -> bool:
    return os.environ.get("AMLT_OUTPUT_DIR", None) is not None




def train(args: argparse.Namespace) -> None:

    # get the config, tokenizer, and model
    torch.cuda.set_device(args.gpu)
    DEVICE = torch.device('cuda:%d' % args.gpu)
    output_dir = os.path.join(args.out_fpath, args.landscape, "splits")
    os.makedirs(output_dir, exist_ok=True)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    model = model.to(DEVICE).eval()

    seq_to_result = {}
    cache_file = os.path.join(output_dir, "esm2_650m.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            seq_to_result = json.load(f)

    # Get files
    ## Grab data
    landscape_path = os.path.join(args.data_fpath, args.landscape, "splits")
    split_csvs = os.listdir(landscape_path)
    split_csvs = [csv for csv in split_csvs if ".csv" in csv]
    print(split_csvs)

    for split_csv in split_csvs:
        df = pd.read_csv(os.path.join(landscape_path, split_csv))
        likelihoods = np.empty(len(df))
        for row in tqdm(df.itertuples(), total=len(df)):
            sequence = row.sequence
            if sequence not in seq_to_result:
                batch_labels, batch_strs, batch_tokens = batch_converter([("seq", sequence)])

                with torch.no_grad():
                    logits = model(batch_tokens.to(DEVICE))["logits"][0, 1:-1]
                    out = F.cross_entropy(logits, batch_tokens[0, 1:-1].to(DEVICE))
                    seq_to_result[sequence] = -out.detach().cpu().item()
                likelihoods[row.Index] = seq_to_result[sequence]
        with open(cache_file, "w") as f:
            json.dump(seq_to_result, f)
        df['esm2_650M_scores'] = likelihoods
        df.to_csv(os.path.join(output_dir, split_csv), index=False)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_fpath', type=str)
    parser.add_argument('out_fpath', type=str)
    parser.add_argument('landscape', type=str)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument("--no_fa2", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()





