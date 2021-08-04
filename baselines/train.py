# set device
# device = cuda... argparse (different GPUs)
# move things manually 

#gpu1 = torch.device('cuda:0') # ADD TO ARGPARSE


import torch
import torch.nn as nn
import numpy as np 



def train_linear_closed_form(train_x, train_y, model):

    # solve the systems of equations
    lin_weights = model(train_x, train_y, training=True)
    print('done fitting least squares')
    
    return lin_weights.solution

def train_linear(train_iterator, val_iterator, device, model, optimizer, r_b, epoch_num):

    
    if r_b == 'r':
        criterion = nn.MSELoss()
    if r_b == 'b':
        criterion = nn.BCEWithLogitsLoss()
    
    val_loss = []

    model.to(device)

    for epoch in range(epoch_num):

        correct = 0

        for i, data in enumerate(train_iterator):

            optimizer.zero_grad()

            inputs, labels = data 
            #input_mask = (inputs != pad_index).long().unsqueeze(-1) # 0 where padding tokens are
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            #outputs = byte_attn(inputs, input_mask)
            outputs = model(inputs).squeeze()

            #loss = criterion(outputs, labels.unsqueeze(-1)).float()
            loss = criterion(outputs, labels)
            loss.backward() 

            optimizer.step()
            
            if r_b == 'b': 
                correct += (torch.round(torch.sigmoid(outputs)) == labels).sum()
        
        
        #if r_b == 'b': 
            #accuracy = correct / len(size)

        with torch.no_grad(): # evaluate validation loss here 

            val_loss_epochs = []
            val_correct = 0

            for data in val_iterator:
                inputs, labels = data 
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                outputs = model(inputs).squeeze()

                loss = criterion(outputs, labels)
                val_loss_epochs.append(loss.item())
                if r_b == 'b': 
                    val_correct += (torch.round(torch.sigmoid(outputs)) == labels).sum()
        
            val_loss_epoch = np.mean(val_loss_epochs)
            val_loss.append(round(val_loss_epoch, 3))
            #if r_b == 'b': 
                #val_accuracy = val_correct / len(train_size)
    
        #if r_b == 'b': 
            #print('epoch: %d loss: %.3f acc: %.2f val loss: %.3f val acc: %.2f' % (epoch + 1, loss.item(), accuracy, val_loss_epoch, val_accuracy))
        if r_b == 'r':
            print('epoch: %d loss: %.3f val loss: %.3f' % (epoch + 1, loss.item(), val_loss_epoch))
       
        # evalutate whether validation loss has reached a new low in last 3 epochs

        if epoch > 21:
            if val_loss[-1] >= np.min(val_loss[:-20]): 
                print('Validation loss no longer going down - finished training')
                break