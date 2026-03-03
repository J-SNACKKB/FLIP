import argparse
import os
from typing import Tuple
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sequence_models.constants import START, STOP

def is_amlt() -> bool:
    return os.environ.get("AMLT_OUTPUT_DIR", None) is not None


class SimpleCollator():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, seq: "str") -> Tuple[torch.Tensor]:
        fwd = START + seq + STOP
        bwd = STOP + seq[::-1] + START
        tokenized = self.tokenizer([fwd, bwd], return_tensors="pt", return_token_type_ids=False)
        return (tokenized['input_ids'],)



def train(args: argparse.Namespace) -> None:

    # get the config, tokenizer, and model
    torch.cuda.set_device(args.gpu)
    DEVICE = torch.device('cuda:%d' % args.gpu)
    output_dir = os.path.join(args.out_fpath, args.landscape, "splits")
    os.makedirs(output_dir, exist_ok=True)
    model_names = [
        ['microsoft/Dayhoff-3b-UR90', "3b-UR90-seq_to_result.json", "dayhoff_3bur90"],
        ['microsoft/Dayhoff-3b-GR-HM-c', "3b-gr-hm-c-seq_to_result.json", "dayhoff"]
    ]
    models = []
    for m in model_names:
        model = AutoModelForCausalLM.from_pretrained(m[0])
        tokenizer = AutoTokenizer.from_pretrained(m[0], trust_remote_code=True)

        # model = AutoModelForCausalLM.from_pretrained('microsoft/Dayhoff-3b-GR-HM-c')
        # tokenizer = AutoTokenizer.from_pretrained('microsoft/Dayhoff-3b-GR-HM-c', trust_remote_code=True)
        collator = SimpleCollator(tokenizer)

        # Move only model to GPU
        model = model.to(DEVICE)
        model = model.to(torch.bfloat16)
        model = model.eval()
        seq_to_result = {}
        cache_file = os.path.join(output_dir, m[1])
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                seq_to_result = json.load(f)
        models.append([model, tokenizer, collator, cache_file, seq_to_result, m[2]])

    # Get files
    ## Grab data
    batch_size = 1
    landscape_path = os.path.join(args.data_fpath, args.landscape, "splits")
    split_csvs = os.listdir(landscape_path)
    split_csvs = [csv for csv in split_csvs if ".csv" in csv]
    print(split_csvs)


    for split_csv in split_csvs:
        df = pd.read_csv(os.path.join(landscape_path, split_csv))
        for model in models:
            model, tokenizer, collator, cache_file, seq_to_result, model_stem = model
            if args.landscape == "PDZ3":
                fwd_lls = [np.empty(len(df)), np.empty(len(df))]
                bwd_lls = [np.empty(len(df)), np.empty(len(df))]
            else:
                fwd_lls = [np.empty(len(df))]
                bwd_lls = [np.empty(len(df))]
            for row in tqdm(df.itertuples(), total=len(df)):
                sequence = row.sequence
                if ":" in sequence:
                    sequences = sequence.split(":")
                else:
                    sequences = [sequence]

                for j, sequence in enumerate(sequences):
                    if sequence not in seq_to_result:
                        tokenized = collator(sequence)[0]
                        tokenized = tokenized.to(DEVICE)
                        with torch.no_grad():
                            out = model(input_ids=tokenized[:1], labels=tokenized[:1])
                            seq_to_result[sequence] = {"fwd": out.loss.detach().cpu().item()}
                        with torch.no_grad():
                            out = model(input_ids=tokenized[1:], labels=tokenized[1:])
                            seq_to_result[sequence]["bwd"] = out.loss.detach().cpu().item()

                    fwd_lls[j][row.Index] = seq_to_result[sequence]["fwd"]
                    bwd_lls[j][row.Index] = seq_to_result[sequence]["bwd"]
            with open(cache_file, "w") as f:
                json.dump(seq_to_result, f)
            if args.landscape == "PDZ3":
                df[model_stem + '_fwd_1'] = -fwd_lls[0]
                df[model_stem + '_bwd_1'] = -bwd_lls[0]
                df[model_stem + '_fwd_2'] = -fwd_lls[1]
                df[model_stem + '_bwd_2'] = -bwd_lls[1]
            else:
                df[model_stem + '_fwd'] = -fwd_lls[0]
                df[model_stem + '_bwd'] = -bwd_lls[0]
                df[model_stem + '_min'] = -np.maximum(fwd_lls, bwd_lls)
        df.to_csv(os.path.join(output_dir, split_csv), index=False)



        if args.landscape != "PDZ3":
            if "esm2_650M_scores" in df.columns:
                print(split_csv,
                      spearmanr(df[df['set'] == 'test']['target'], df[df['set'] == 'test']['dayhoff_3bur90_min']).statistic,
                      spearmanr(df[df['set'] == 'test']['target'], df[df['set'] == 'test']['dayhoff_min']).statistic,
                      spearmanr(df[df['set'] == 'test']['target'], df[df['set'] == 'test']['esm2_650M_scores']).statistic)
            else:
                print(split_csv,
                      spearmanr(df[df['set'] == 'test']['target'], df[df['set'] == 'test']['dayhoff_3bur90_min']).statistic,
                      spearmanr(df[df['set'] == 'test']['target'], df[df['set'] == 'test']['dayhoff_min']).statistic)



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





