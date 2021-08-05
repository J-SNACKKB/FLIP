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


def regression_eval(predicted, labels, print_stats, SAVE_PATH):
    """ outputs spearman, MSE, and graph of predicted vs actual """
    
    if type(predicted) == list: 
        predicted = concat_tensor(predicted)
        labels = concat_tensor(labels)
    else: 
        predicted = np.array(predicted)
        labels = np.array(labels)

    rho, _ = stats.spearmanr(predicted, labels) # spearman
    mse = mean_squared_error(predicted, labels) # MSE
    
    if print_stats:
        print('stats: Spearman: %.2f MSE: %.2f ' % (rho, mse))

        plt.figure()
        plt.title('predicted (y) vs. labels (x)')
        sns.scatterplot(x = labels, y = predicted, s = 2, alpha = 0.2)
        plt.show()
    
    else:
        print('train stats: Spearman: %.2f MSE: %.2f ' % (rho, mse))
        plt.figure()
        plt.title('predicted (y) vs. labels (x)')
        sns.scatterplot(x = labels, y = predicted, s = 2, alpha = 0.2)
        plt.savefig(SAVE_PATH+'.png', dpi = 300)
        
 ##### TODO: SAVE_PATH
        
    
def binary_eval(predicted, labels, scores, print_stats):
    """ outputs ROC plot, average precision/recall """
    
    if type(predicted) == list: 
        predicted = concat_tensor(predicted)
        labels = concat_tensor(labels)
        scores = concat_tensor(scores)
    else: 
        predicted = np.array(predicted)
        labels = np.array(labels)
        scores = np.array(scores)

    if print_stats:
        # accuracy
        acc = metrics.accuracy_score(predicted, labels)
        print("Accuracy: %.2f" % acc)

        # Precision & Recall
        average_precision = metrics.average_precision_score(labels, scores)

        print('Average precision-recall score: {0:0.2f}'.format(
              average_precision))

        # ROC
        fpr, tpr, trhesholds = metrics.roc_curve(labels, scores, pos_label = 1)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = roc_auc)

        display.plot() 
        plt.show() 


def evaluate_linear_closed_form(x, y, model, solution):
    
    model.eval()
    
    # get stats on training set
    x = model(x, y = None, training = False) # one-hot encode x
    predicted_y = torch.matmul(x, solution.double()).squeeze()
    #predicted_labels = torch.round(predicted_y)

    # evaluate
    print('stats:')
    #binary_eval(predicted = predicted_labels, labels = y, scores = predicted_y)
    regression_eval(predicted = predicted_y, labels = y)
    
def evaluate_linear(data_iterator, model, device, r_b, SAVE_PATH):
    """ run data through model and print eval stats """
    scores = [] # create lists to hold tensors created each minibatch
    predicted = []
    labels_ls = []
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():

        for data in data_iterator:
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs).squeeze() # Forward prop without storing gradients
            
            if r_b == 'b':
                scores.append(torch.sigmoid(outputs))
                predicted.append(torch.round(torch.sigmoid(outputs))) 
            if r_b == 'r':
                predicted.append(outputs.to('cpu'))
            
            labels_ls.append(labels.to('cpu'))

    #print('stats:')
    if r_b == 'b':
        binary_eval(predicted = predicted, labels = labels_ls, scores = scores, )
    if r_b == 'r':
        regression_eval(predicted = predicted, labels = labels_ls, print_stats = False, SAVE_PATH = SAVE_PATH)
        
    with open(SAVE_PATH+'.pickle', 'wb') as f:
        pickle.dump((predicted, labels_ls), f)

def evaluate_esm(data_iterator, device, model, size, SAVE_PATH):
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
            m = (inp[:, :, 0] != 0).long().to(device)

            o = model(inp, m).squeeze().cpu()  # Forward prop without storing gradients

            b = inp.shape[0] 
            out[s: s + b:] = o
            labels[s: s + b:] = l

            s += b
    
    print('size of predicted:', out.shape)

    regression_eval(predicted=out, labels=labels, print_stats=False, SAVE_PATH=SAVE_PATH)

    with open(SAVE_PATH+'.pickle', 'wb') as f:
        pickle.dump((out, labels), f)