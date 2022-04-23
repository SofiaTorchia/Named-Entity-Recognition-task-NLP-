import torch.nn as nn

class SkipGram(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, id2word):
        super(SkipGram, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Linear(self.vocabulary_size, self.embedding_dim)
        self.output_weights = nn.Linear(self.embedding_dim, self.vocabulary_size)
        self.loss_function = nn.CrossEntropyLoss()


    def forward(self, input_idx):
        input_embeddings = self.embeddings(input_idx)  
        output_embeddings = self.output_weights(input_embeddings)
        output = output_embeddings
        return output