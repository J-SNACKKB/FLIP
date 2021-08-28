from sklearn import metrics
from sklearn.metrics import mean_squared_error
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def concat_tensor(tensor_list, keep_tensor = False):
    """ converts a list of tensors to a numpy array for stats analysis """
    for i, item in enumerate(tensor_list):
        item.to('cpu')
        if i == 0:
            output_tensor = item
        if i > 0:
            output_tensor = torch.cat((output_tensor, item), 0)
    
    if keep_tensor:
        return output_tensor
    else:
        return np.array(output_tensor)

def regression_eval(predicted, labels, SAVE_PATH):
    """ 
    input: 1D tensor or array of predicted values and labels
    output: saves spearman, MSE, and graph of predicted vs actual 
    """

    predicted = np.array(predicted)
    labels = np.array(labels)

    rho, _ = stats.spearmanr(predicted, labels) # spearman
    mse = mean_squared_error(predicted, labels) # MSE

    plt.figure()
    plt.title('predicted (y) vs. labels (x)')
    sns.scatterplot(x = labels, y = predicted, s = 2, alpha = 0.2)
    plt.savefig(SAVE_PATH / 'preds_vs_labels.png', dpi = 300)

    return round(rho, 2), round(mse, 2)

def evaluate_esm(data_iterator, model, device, size, mean, mut_mean, SAVE_PATH):
    """ run data through model and print eval stats """
    
    # create a tensor to hold results
    out = np.empty([size])
    labels = np.empty([size])

    s = 0 
    
    model.eval()
    model.to(device)

    with torch.no_grad(): # evaluate validation loss here 
        for i, (inp, l) in enumerate(data_iterator):
            
            inp = inp.to(device)

            if mean or mut_mean: 
                o = model(inp).squeeze().cpu()
            else:
                m = (inp[:, :, 0] != 0).long().to(device)
                o = model(inp, m).squeeze().cpu()  # Forward prop without storing gradients

            b = inp.shape[0] 
            out[s: s + b:] = o
            labels[s: s + b:] = l

            s += b

    if mean:
        SAVE_PATH = SAVE_PATH / 'mean'
    if mut_mean:
        SAVE_PATH = SAVE_PATH / 'mut_mean'
        
    SAVE_PATH.mkdir(parents=True, exist_ok=True) # make directory if it doesn't exist already
    with open(SAVE_PATH / 'preds_labels_raw.pickle', 'wb') as f:
        pickle.dump((out, labels), f)
    
    rho, mse = regression_eval(predicted=out, labels=labels, SAVE_PATH=SAVE_PATH)

    return rho, mse
