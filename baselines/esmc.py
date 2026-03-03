import argparse
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from sequence_models.structure import Attention1d
from sequence_models.utils import warmup
from sequence_models.flip_utils import load_flip_data





class Model(nn.Module):

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.attention = Attention1d(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, e, input_mask=None):
        attended = self.attention(e, input_mask=input_mask)
        hidden = self.hidden(self.activation(attended))
        return self.linear(self.dropout(self.activation(hidden)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_fpath', type=str)
    parser.add_argument('out_fpath', type=str)
    parser.add_argument('task', type=str)
    parser.add_argument('weights', type=str, default='pretrained')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    args = parser.parse_args()
    train(args)


def train(args):
    _ = torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    lr = args.lr

    np.random.seed(args.seed)
    esm = ESMC.from_pretrained("esmc_300m") # or "cpu"
    if args.weights != 'pretrained':
        for p in esm.modules():
            try:
                p.reset_parameters()
            except AttributeError:
                continue
    esm = esm.to(device)
    tokenizer = EsmSequenceTokenizer()

    def collator(batch):
        data = tuple(zip(*batch))
        seqs, labels = data
        t = [torch.tensor(tokenizer.encode(s)) for s in seqs]
        max_len = max([len(tt) for tt in t])
        t = [F.pad(tt, (0, max_len - len(tt)), value=1) for tt in t]
        t = torch.stack(t)
        y = torch.tensor(labels).unsqueeze(-1).float()
        input_mask = t != 1
        return t, y, input_mask


    d_model = 960
    decoder = Model(d_model, dropout=0).to(device)
    model = nn.ModuleDict({'embed': esm.embed, 'transformer': esm.transformer, 'decoder': decoder})
    optimizer = Adam(model.parameters(), lr=lr)


    ## Grab data
    batch_size = 16
    loss_func = nn.MSELoss()
    if "AMY_BACSU" in args.task:
        flip_dataset = '_'.join(args.task.split('_')[:2])
        flip_split = '_'.join(args.task.split('_')[2:])
    else:
        flip_dataset = args.task.split('_')[0]
        flip_split = '_'.join(args.task.split('_')[1:])
    ds_train, ds_valid, ds_test = load_flip_data(args.data_fpath, flip_dataset, flip_split, max_len=2048, scale=True)
    num_workers = 4
    dl_train = DataLoader(ds_train, batch_size=batch_size, collate_fn=collator,
                              num_workers=num_workers, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)
    print('%d Train samples %d valid samples %d test samples' %(len(ds_train), len(ds_valid), len(ds_test)))
    checkpoint_stem = 'esmc_%s_%s_%d' %(args.task, args.weights, args.seed)
    def step(model, batch, train=True, return_values=False):
        src, tgt, input_mask = batch
        src = src.to(device)
        tgt = tgt.to(device)
        input_mask = input_mask.to(device)
        e = model['embed'](src)
        e = model['transformer'](e)[0].float()
        outputs = model['decoder'](e, input_mask=input_mask)

        loss = loss_func(outputs, tgt)
        locations = len(tgt)
        mask = torch.ones(1)  # dummy
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if return_values:
            return loss.item(), locations, outputs.detach().cpu(), src.detach().cpu(), tgt.detach().cpu(), mask.detach().cpu()
        else:
            return loss.item(), locations


    def epoch(model, current_step=0):
        model = model.train()
        loader = dl_train
        t = 'Training:'
        losses = []
        ns = []
        n_seen = 0
        if train:
            n_total = len(ds_train)
        else:
            n_total = len(ds_valid)
        for i, batch in enumerate(loader):
            new_loss, new_n = step(model, batch, True)
            losses.append(new_loss * new_n)
            ns.append(new_n)
            n_seen += len(batch[0])
            total_n = sum(ns)
            if total_n == 0:
                rloss = 0
            else:
                rloss = sum(losses) / total_n
            if train:
                nsteps = current_step + i + 1
            else:
                nsteps = i
            print('\r%s Epoch %d of %d Step %d Example %d of %d loss = %f'
                  % (t, e + 1, epochs, nsteps, n_seen, n_total, rloss),
                  end='')
        if not train:
            return rloss
        return i, rloss

    def test_epoch(model, dl):
        model = model.eval()
        with torch.no_grad():
            losses = []
            ns = []
            n_seen = 0
            pred = []
            tgt = []
            masks = []
            for i, batch in enumerate(dl):
                new_loss, new_n, p, s, t, m = step(model, batch, False, return_values=True)
                losses.append(new_loss * new_n)
                pred.append(p)
                tgt.append(t)
                masks.append(m)
                ns.append(new_n)
                n_seen += len(batch[0])
                total_n = sum(ns)

        test_loss = sum(losses) / total_n
        pred = torch.cat(pred)
        tgt = torch.cat(tgt)
        pred = pred.numpy()
        tgt = tgt.numpy()
        spearman = spearmanr(pred, tgt).correlation
        if (tgt < 0).any():
            pos_tgt = tgt - tgt.min()
        else:
            pos_tgt = tgt
        ndcg = ndcg_score(pos_tgt.T, pred.T)
        print('\tloss: %f' %test_loss, end='\t')
        print('spearman: %f' %(spearman), end='\t')
        print('ndcg: %f' %(ndcg), end='\t')
        results = {
            'spearman': spearman,
            'loss': test_loss,
            'ndcg': ndcg
        }
        return results

    epochs = 500
    n_warmup = 1000
    total_steps = 0
    best_valid_metric = -np.inf
    best_valid_loss = np.inf
    patience = 10
    scheduler = LambdaLR(optimizer, warmup(n_warmup))
    waiting = 0
    os.makedirs(args.out_fpath, exist_ok=True)
    for e in range(epochs):
        ts, train_loss = epoch(model, current_step=total_steps)
        total_steps += ts
        nsteps = total_steps
        results = test_epoch(model, dl_valid)
        vloss = results['loss']
        vmetric = results['spearman']
        waiting += 1
        if vloss < best_valid_loss:
            best_valid_loss = vloss
            waiting = 0
            torch.save({
                'step': nsteps,
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'val_spearman': vmetric,
                'val_ndcg': results['ndcg'],
                'val_loss': vloss,
                'train_loss': train_loss,
            }, args.out_fpath + checkpoint_stem + '_best.pt')
        if vmetric > best_valid_metric:
            best_valid_metric = vmetric
            waiting = 0
        if vloss < train_loss:
            waiting = 0
        print("waiting: %d" % waiting)
        if waiting == patience:
            break
    # TODO: checkpoint race condition
    if args.out_fpath is not None:
        sd = torch.load(args.out_fpath + checkpoint_stem + '_best.pt', weights_only=False)
        model.load_state_dict(sd['model_state_dict'])
        results = test_epoch(model, dl_test)
        results['batch_size'] = batch_size
        results['lr'] = lr
        results['epoch'] = sd['epoch']
        results['step'] = sd['step']
        results['train_loss'] = sd['train_loss']
        results['val_spearman'] = sd['val_spearman']
        results['val_loss'] = sd['val_loss']
        results['val_ndcg'] = sd['val_ndcg']
        results['dataset'] = flip_dataset
        results['split'] = flip_split
        results['task'] = args.task
        results['seed'] = args.seed
        with open(args.out_fpath + checkpoint_stem + '.json', 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main()