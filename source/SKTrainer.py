from tqdm import tqdm
import torch
import os

class SKTrainer():
    '''
    This class implments a trainer for a Skipgram model 
    '''
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.model.train()

    def train(self, train_dataset, output_folder, vocab_size, epochs=1):
        train_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            len_train = 0

            for step, sample in enumerate(train_dataset):
                inputs = sample['inputs']
                targets = sample['outputs']
                one_hot_input = torch.zeros((inputs.shape[0], vocab_size))
                for i, x in enumerate(inputs):
                    one_hot_input[i, x] = 1

                output_distribution = self.model(one_hot_input)
                loss = self.model.loss_function(output_distribution, targets) 
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()
                len_train += 1

            avg_epoch_loss = epoch_loss / len_train
            #print('Epoch: {}, avg epoch loss = {:0.4f}'.format(epoch, avg_epoch_loss))
            train_loss += avg_epoch_loss
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss

    
