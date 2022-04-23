import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import collections
import torch


class MakeDataset(Dataset):

    '''
        This class creates 
            - a vocabulary of the ("vocab_size" - 1) most frequent words in a dataset ("word2id" and "id2word")
            - a label vocabulary ("label2id" and "id2label")
            - creates a new dataset consisting of subsequences (of the original sentences) of length 
                "window_size" shifting the window of "window_shift" tokens
        (Code partially taken from notebooks)
    '''
    
    def __init__(self, data_path, window_size, window_shift, stop_token, unk_token, pad_token, vocab_size = 10000):
        self.stop_token = stop_token
        self.pad_token = pad_token
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.window_size = window_size
        self.window_shift = window_shift
        self.data_path = data_path

        raw_data = pd.read_csv(data_path, sep='\t', header=None)   # transforms the input file in a DataFrame
        raw_data = raw_data.rename(columns = {0:'token',1:'label',2:'id'})
        self.raw_data = raw_data

        # first, read the original data and create the single complete sentences using "create_sentences"
        # second, create subsequences for each osiginal sentence with "create_windows"
        sentences = self.create_sentences(raw_data)  
        self.windows = self.create_windows(sentences)

        self.sentences = sentences
        self.word2id = self.build_vocab(raw_data)
        self.id2word = {value:key for key,value in self.word2id.items()}

        self.label2id = self.build_label_vocab(raw_data)
        self.id2label = {value:key for key,value in self.label2id.items()}
    


    def create_sentences(self,raw_data):
        sentences = []
        self.tok_sentences = []
        cond = raw_data['token'].apply(lambda x: x == self.stop_token)
        stop_set = raw_data[cond].index.tolist()
        for i in range(len(stop_set)-1):
            sentence = raw_data.iloc[stop_set[i]+1:stop_set[i+1]]
            self.tok_sentences.append(list(sentence['token']))  
            d = [{'form':form,'label':label} for form,label in zip(sentence['token'],sentence['label'])]
            sentences.append(d)
        return sentences
    


    def create_windows(self,sentences):  
        windows = []
        for sentence in sentences:
            for i in range(0,len(sentence), self.window_shift):
                window = sentence[i:i+self.window_size]
                if len(window) < self.window_size: 
                    window = window + [None]*(self.window_size - len(window)) # Adding padding when necessary
                assert len(window) == self.window_size
                windows.append(window)
        return windows



    def build_vocab(self, raw_data):
        counter_list = []
        for token in raw_data['token']:
            if token != self.stop_token and token != self.unk_token and token != self.pad_token:
                counter_list.append(token)
        counter = collections.Counter(counter_list)
        dictionary = {key: index for index, (key, _) in enumerate(counter.most_common(self.vocab_size - 2))}
        assert self.unk_token not in dictionary
        assert self.pad_token not in dictionary
        dictionary[self.unk_token] = self.vocab_size - 2
        dictionary[self.pad_token] = self.vocab_size - 1

        dict_counts = {x: counter[x] for x in dictionary if (x is not self.unk_token and x is not self.pad_token)}
        self.frequency = dict_counts
        self.tot_occurrences = sum(dict_counts[x] for x in dict_counts)
        return dictionary



    def build_label_vocab(self, raw_data):
        counter_list = []
        for label in raw_data['label']:
            if label and label != 'id':
                counter_list.append(label)
        counter = collections.Counter(counter_list)
        dictionary = {key: index for index, (key, _) in enumerate(counter.most_common())}
        dictionary[self.pad_token] = len(counter)
        return dictionary



    def make_dataset(self, path):
        dataset = []
        raw_data = pd.read_csv(path, sep='\t', header=None)
        raw_data = raw_data.rename(columns = {0:'token',1:'label',2:'id'})
        windows = self.create_windows(self.create_sentences(raw_data))
        for window in windows:
            encoded_tokens = torch.LongTensor(self.encode_text(window))  # transforms tokens in their id value in the vocabulary ("word2id")
            encoded_labels = torch.LongTensor(self.encode_label(window)) # transforms labels in their id value in the label vocabulary ("label2id")
            dataset.append({'inputs':encoded_tokens,'outputs':encoded_labels})
        return dataset



    def encode_label(self,window):
        indices = []
        for element in window:
            if element is None:
                indices.append(self.label2id[self.pad_token]) 
            else:
                indices.append(self.label2id[element['label']])
        return indices



    def encode_text(self,window):
        indices = []
        for element in window:
            if element is None:
                indices.append(self.word2id[self.pad_token]) 
            elif element['form'] in self.word2id.keys():
                indices.append(self.word2id[element['form']])
            else:
                indices.append(self.word2id[self.unk_token])
        return indices
    





