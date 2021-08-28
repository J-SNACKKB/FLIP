import torch
import torch.nn as nn
import numpy as np 
import sys 

def train_esm(train_iterator, val_iterator, model, device, criterion, optimizer, epoch_num, mean):
    
        
        val_loss = []
        
        model.to(device)

        for epoch in range(epoch_num):
            model.train() 
            for i, (inp, l) in enumerate(train_iterator):
                optimizer.zero_grad()
                
                inp = inp.to(device)
                l = l.to(device) 

                if mean:
                    o = model(inp)
                else:
                    m = (inp[:, :, 0] != 0).long().to(device)
                    o = model(inp, m) 

                loss = criterion(o, l.unsqueeze(-1))
                loss.backward() 
                optimizer.step()

            with torch.no_grad(): # evaluate validation loss here 

                model.eval()
                val_loss_epochs = []

                for (inp, l) in val_iterator:
                    
                    inp = inp.to(device)
                    l = l.to(device)

                    if mean:
                        o = model(inp)
                    else:
                        m = (inp[:, :, 0] != 0).long().to(device)
                        o = model(inp, m)  # Forward prop without storing gradients

                    loss = criterion(o, l.unsqueeze(-1)) # Calculate validation loss 
                    val_loss_epochs.append(loss.item())

                val_loss_epoch = np.mean(val_loss_epochs)
                val_loss.append(round(val_loss_epoch, 3))

            print('epoch: %d loss: %.3f val loss: %.3f' % (epoch + 1, loss.item(), val_loss_epoch))

            # evalutate whether validation loss is dropping; if not, stop
            if epoch > 21:
                if val_loss[-1] >= np.min(val_loss[:-20]): 
                    print('Finished training at epoch {0}'.format(epoch))
                    return epoch 
    

            
                    
