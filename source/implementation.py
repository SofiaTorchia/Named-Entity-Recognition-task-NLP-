import numpy as np
from typing import List, Tuple
from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel()
    #return RandomBaseline()


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]

import pickle
import torch
import stud.NERTaggerModel as pt
import stud.MakeDataset as md
import stud.SkipGram as sk
rootpath = './'        

class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
   
    def __init__(self):
        with open(rootpath + 'model/Output folder/data.pickle', 'rb') as file:
            data = pickle.load(file)

        emb_path = f'model/Output folder/word_embedding.pt'
        model_path = f'model/Output folder/state_BiLSTM_CRF_W2V.pt'
        sk_model = sk.SkipGram(15000, 200, data.id2word)
        sk_model.load_state_dict(torch.load(rootpath + emb_path))
        weights = sk_model.state_dict()['output_weights.weight']

        class HParams():
            vocab_size = 15000
            hidden_dim = 50
            embedding_dim = 200
            num_classes = 14
            bidirectional = True
            num_layers = 1
            dropout = 0.2
            embeddings = weights
            crf = True
            loss_function = torch.nn.CrossEntropyLoss()
        self.model_params = HParams()

        self.word2id = data.word2id    # token-dictionary
        self.id2label = data.id2label   # label-dictionary
        self.unk_token = data.unk_token
        self.model = pt.NERTaggerModel(self.model_params)
        self.model.load_state_dict(torch.load(rootpath + model_path))
        return
 

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        self.model.eval()
        encoded_tokens = self.encode_text(tokens)
        outputs = []
        for enc_token in encoded_tokens:
            enc_token = torch.LongTensor([enc_token])
            with torch.no_grad():
                logits = self.model(enc_token,predict = True)
                output = self.decode_output(logits[0])
            outputs.append(output)
        return outputs
    

    def encode_text(self,tokens):
        indices_list = []
        for token_list in tokens:
            indices = []
            for token in token_list:
                if token in self.word2id.keys():
                    indices.append(self.word2id[token])
                else:
                    indices.append(self.word2id[self.unk_token])
            indices_list.append(indices)
        return indices_list
    

    def decode_output(self,logits):
        predictions = []
        for indices in logits:
            predictions.append(self.id2label[indices])
        return predictions

    

    def build_data(window_size, window_shift, stop_token, unk_token, pad_token, vocab_size):
        path = rootpath + '../data/train.tsv'
        data = md.MakeDataset(path, window_size, window_shift, stop_token, unk_token, pad_token, vocab_size)
        with open(rootpath + '../model/Output folder/data.pickle', 'wb') as file:
            pickle.dump(data, file)
        
        