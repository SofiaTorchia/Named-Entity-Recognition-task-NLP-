import torch.nn as nn
import torch
from  torchcrf import CRF
import numpy as np

class NERTaggerModel(nn.Module):
    '''
    This class implements a Bi-LSTM CRF model for NER tagging tasks
    '''
    
    def __init__(self, hparams):
        super(NERTaggerModel, self).__init__()
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        if hparams.embeddings is not None:
            self.word_embedding.weight.data.copy_(hparams.embeddings)
        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim, bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, dropout = hparams.dropout if hparams.num_layers > 1 else 0)
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)
        self.loss_function = hparams.loss_function
        self.is_crf = hparams.crf
        if hparams.crf == True:
            self.crf = CRF(hparams.num_classes,batch_first = True)

    
    def forward(self, x, tags=None, first=False, eval=False, predict=False):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)

        if self.is_crf:
            if tags is not None:
                mask = self.create_mask(tags, output, first)  # mask tensor for padding
                l = self.crf(output,tags,mask)  # returns log likelihood of a sequence of tags given emission scores
                return -l
            else:
                return self.crf.decode(output)  # predicts labels for output

        if self.is_crf == False:
            if tags is not None:
                output = output.view(-1, output.shape[-1])
                tags = torch.LongTensor(tags)
                tags = tags.view(-1)
                sample_loss = self.loss_function(output, tags) 
                return sample_loss
            else:
                output = torch.argmax(output,-1).tolist()
                return output


    def create_mask(self, tags, output, first):
        tags = torch.LongTensor(tags)
        mask = [[1 for i in range(tags.size(1))]] * tags.size(0)
        mask = np.array(mask)
        if first == False: # for the first epoch mask must not have zeros
            for i in range(tags.size(0)):
                for j in range(tags.size(1)):
                    if tags[i,j]>12:   # mask only the padding tag
                        mask[i,j] = 0
        mask = torch.ByteTensor(mask) 
        return mask

            

