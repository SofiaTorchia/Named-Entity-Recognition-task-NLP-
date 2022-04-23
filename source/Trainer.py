import torch.nn as nn
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import confusion_matrix


class Trainer(): 
    '''
    This class implements a trainer for the NERTagger model 
    with the possibility of early stopping.
    '''
    def __init__(self, model: nn.Module, optimizer, label_vocab):
        self.model = model
        self.optimizer = optimizer
        self.label_vocab = label_vocab


    def train(self, train_dataset:Dataset, valid_dataset:Dataset, output_folder:str, epochs:int=1, 
                early_stopping:bool=False, early_stopping_patience:int=1):
        train_history = [] 
        valid_history = [] 
        patience_counter = 0
        assert epochs > 1 and isinstance(epochs, int)

        for epoch in tqdm(range(epochs)):
            losses = [] 
            self.model.train()

            for step, sample in enumerate(train_dataset):
                inputs = sample['inputs']
                labels = sample['outputs']
                self.optimizer.zero_grad()
                sample_loss = self.model(inputs,labels,first = (epoch==1))      # returns the loss computed between 
                                            # the the output predicted for the current input and the correct labels
                sample_loss.backward()
                self.optimizer.step()
                losses.append(sample_loss.tolist()) 

            mean_loss = sum(losses)/len(losses) 
            train_history.append(mean_loss) 

            valid_loss = self.evaluate(valid_dataset)
            valid_history.append(valid_loss) 

            if early_stopping:
                if epoch>0 and valid_history[-1]>valid_history[-2]:
                    print(f'Validation error increases for epoch {epoch}')
                    if patience_counter >= early_stopping_patience:
                        print('patience_counter >= early_stopping_patience')
                        print(f'Current epoch: {epoch}')
                        torch.save(self.model.state_dict(),os.path.join(output_folder,'state.pt'))
                        break
                    else:
                        patience_counter += 1
        return {'train_history': train_history, 'valid_history': valid_history, 'current_epoch': epoch}


    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        self.model.eval()   
        with torch.no_grad():
            for sample in valid_dataset:
                inputs = sample['inputs']
                labels = sample['outputs']
                predictions = self.model(inputs,labels, eval = True)
                sample_loss = predictions
                valid_loss += sample_loss.tolist()
        return valid_loss / len(valid_dataset)



    def confusion_matrix(self, valid_dataset):
        self.model.eval()   
        y_pred = []
        y_true = []

        with torch.no_grad():
            for sample in valid_dataset:

                inputs = sample['inputs']
                labels = sample['outputs']

                predictions = self.model(inputs)
                p = np.array(predictions)
                p = p.ravel()
                l = np.array(labels)
                #print(l.shape)
                l = l.ravel()
                #print(p.shape)
                #print(labels.tolist())
                y_pred.extend(p)
                y_true.extend(l)
        m = confusion_matrix(y_true,y_pred)
        return m
        
    