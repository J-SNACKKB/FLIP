from typing import List, Any
from datetime import datetime
import argparse
from scipy.stats import spearmanr
import pandas as pd
from sklearn.metrics import mean_squared_error



import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from train_all import split_dict 
from utils import SequenceDataset, load_dataset
from csv import writer
from pathlib import Path

AAINDEX_ALPHABET = 'ARNDCQEGHILKMFPSTWYVXU'


class Tokenizer(object):
    """Convert between strings and their one-hot representations."""
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    def tokenize(self, seq: str) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in seq])

    def untokenize(self, x) -> str:
        return ''.join([self.t_to_a[t] for t in x])


class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ASCollater(object):
    def __init__(self, alphabet: str, tokenizer: object, pad=False, pad_tok=0., backwards=False):
        self.pad = pad
        self.pad_tok = pad_tok
        self.tokenizer = tokenizer
        self.backwards = backwards
        self.alphabet = alphabet

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        sequences = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        sequences = [i.view(-1,1) for i in sequences]
        maxlen = max([i.shape[0] for i in sequences])
        padded = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]),"constant", self.pad_tok) for i in sequences]
        padded = torch.stack(padded)
        mask = [torch.ones(i.shape[0]) for i in sequences]
        mask = [F.pad(i, (0, maxlen - i.shape[0])) for i in mask]
        mask = torch.stack(mask)
        y = data[1]
        y = torch.tensor(y).unsqueeze(-1)
        ohe = []
        for i in padded:
            i_onehot = torch.FloatTensor(maxlen, len(self.alphabet))
            i_onehot.zero_()
            i_onehot.scatter_(1, i, 1)
            ohe.append(i_onehot)
        padded = torch.stack(ohe)
            
        return padded, y, mask


class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.linear:
            x = F.relu(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x


class FluorescenceModel(nn.Module):
    def __init__(self, n_tokens, kernel_size, input_size, dropout):
        super(FluorescenceModel, self).__init__()
        self.encoder = MaskedConv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(linear=True, in_dim=input_size, out_dim=input_size*2)
        self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout) # TODO: actually add this to model
        self.input_size = input_size

    def forward(self, x, mask):
        # encoder
        x = F.relu(self.encoder(x, input_mask=mask.repeat(self.n_tokens, 1, 1).permute(1, 2, 0)))
        x = x * mask.repeat(self.input_size, 1, 1).permute(1, 2, 0)
        # embed
        x = self.embedding(x)
        x = self.dropout(x)
        # decoder
        output = self.decoder(x)
        return output


def train(args):
    # set up training environment

    batch_size = 256
    epochs = 1000
    device = torch.device('cuda:%d' %args.gpu)
    alphabet = AAINDEX_ALPHABET
    if args.dataset == 'meltome':
        alphabet += 'XU'
        batch_size = 32
    tokenizer = Tokenizer(alphabet)
    print('USING OHE HOT ENCODING')
    criterion = nn.MSELoss()
    model = FluorescenceModel(len(alphabet), args.kernel_size, args.input_size, args.dropout) 
    model = model.to(device)
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-3, 'weight_decay': 0},
        {'params': model.embedding.parameters(), 'lr': 5e-5, 'weight_decay': 0.05},
        {'params': model.decoder.parameters(), 'lr': 5e-6, 'weight_decay': 0.05}
    ])
    patience = 20
    p = 0
    best_rho = -1

    # grab data
    split = split_dict[args.task]
    train, val, test, _ = load_dataset(args.dataset, split+'.csv')

    ds_train = SequenceDataset(train)
    ds_valid = SequenceDataset(val)
    ds_test = SequenceDataset(test)

    # setup dataloaders
    dl_train_AA = DataLoader(ds_train, collate_fn=ASCollater(alphabet, tokenizer, pad=True, pad_tok=0.),
                             batch_size=batch_size, shuffle=True, num_workers=4)
    dl_valid_AA = DataLoader(ds_valid, collate_fn=ASCollater(alphabet, tokenizer, pad=True, pad_tok=0.),
                             batch_size=batch_size, num_workers=4)
    dl_test_AA = DataLoader(ds_test, collate_fn=ASCollater(alphabet, tokenizer, pad=True, pad_tok=0.),
                            batch_size=batch_size, num_workers=4)

    def step(model, batch, train=True):
        src, tgt, mask = batch
        src = src.to(device).float()
        tgt = tgt.to(device).float()
        mask = mask.to(device).float()
        output = model(src, mask)
        loss = criterion(output, tgt)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), output.detach().cpu(), tgt.detach().cpu()


    def epoch(model, train, current_step=0, return_values=False):
        start_time = datetime.now()
        if train:
            model = model.train()
            loader = dl_train_AA
            t = 'Training'
            n_total = len(ds_train)
        else:
            model = model.eval()
            loader = dl_valid_AA
            t = 'Validating'
            n_total = len(ds_valid)
        losses = []
        outputs = []
        tgts = []
        chunk_time = datetime.now()
        n_seen = 0
        for i, batch in enumerate(loader):
            loss, output, tgt = step(model, batch, train)
            losses.append(loss)
            outputs.append(output)
            tgts.append(tgt)
            n_seen += len(batch[0])
            if train:
                nsteps = current_step + i + 1
            else:
                nsteps = i
            #print('\r%s Epoch %d of %d Step %d Example %d of %d loss = %.4f'
                  #% (t, e + 1, epochs, nsteps, n_seen, n_total, np.mean(np.array(losses)),),
                  #end='')
            
        outputs = torch.cat(outputs).numpy()
        tgts = torch.cat(tgts).cpu().numpy()
        if train:
            #print('\nTraining complete in ' + str(datetime.now() - chunk_time))
            with torch.no_grad():
                _, val_rho = epoch(model, False, current_step=nsteps)
            chunk_time = datetime.now()
        if not train:
            #print('\nValidation complete in ' + str(datetime.now() - start_time))
            val_rho = spearmanr(tgts, outputs).correlation
            mse = mean_squared_error(tgts, outputs)

#             print('\nEpoch complete in ' + str(datetime.now() - start_time))
        if return_values:
            return i, mse, val_rho, tgts, outputs
        else:
            return i, val_rho
    nsteps = 0
    e = 0
    bestmodel_name = 'bestmodel_'+str(args.task)+str(args.kernel_size)+'_'+str(args.input_size)+'_'+str(args.dropout)+'.tar'
    for e in range(epochs):
        s, val_rho = epoch(model, True, current_step=nsteps)
        #print(val_rho)
        nsteps += s

        '''
        if (e%10 == 0) or (e == epochs-1):
            torch.save({
                'step': nsteps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, args.out_fpath + 'checkpoint%d.tar' % nsteps)
        '''
        if val_rho > best_rho:
            p = 0
            best_rho = val_rho
            torch.save({
                'step': nsteps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, args.out_fpath + bestmodel_name)
        else:
            p += 1 
        if p == patience:
            print('MET PATIENCE')
            break
    print('Testing...')
    sd = torch.load(args.out_fpath + bestmodel_name)
    model.load_state_dict(sd['model_state_dict'])
    dl_valid_AA = dl_test_AA
    _, mse, val_rho, tgt, pre = epoch(model, False, current_step=nsteps, return_values=True)
    print('rho = %.2f' %val_rho)
    print('mse = %.2f' %mse)
    np.savez_compressed('%s.npz' %args.task, prediction=pre, tgt=tgt)
    with open(Path.cwd() / 'evals_new'/ (args.dataset+'_results.csv'), 'a', newline='') as f:
        writer(f).writerow([args.dataset, 'CNN', split, '', '', '', val_rho, mse, e, args.kernel_size, args.input_size, args.dropout])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='file path to data directory')
    parser.add_argument('task', type=str)
    parser.add_argument('out_fpath', type=str, help='save directory')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ensemble',action='store_true')
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.0)

    args = parser.parse_args()

    if args.ensemble:
        for i in range(10):
            np.random.seed(i)
            torch.manual_seed(i)
            train(args)
    else:
        np.random.seed(10)
        torch.manual_seed(10)
        train(args)

if __name__ == '__main__':
    main()