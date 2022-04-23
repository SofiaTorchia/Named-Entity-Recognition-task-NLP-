import torch
import numpy as np

class Word2VecDataset(torch.utils.data.IterableDataset):

    def __init__(self, tok_sentences, word2id, window_size, frequency, tot_occurrences):
        self.tok_sentences = tok_sentences
        self.word2id = word2id
        self.window_size = window_size
        self.frequency = frequency
        self.tot_occurrences = tot_occurrences



    def __iter__(self):
        for sentence in self.tok_sentences:
            len_sentence = len(sentence)

            for input_idx in range(len_sentence):
                current_word = sentence[input_idx]
                if current_word in self.word2id and self.keep_word(current_word):
                    min_idx = max(0, input_idx - self.window_size)
                    max_idx = min(len_sentence, input_idx + self.window_size)

                    window_idxs = [x for x in range(min_idx, max_idx) if x != input_idx]
                    for target_idx in window_idxs:
                        if sentence[target_idx] in self.word2id:
                            target = self.word2id[sentence[target_idx]]
                            current_word_id = self.word2id[current_word]
                            output_dict = {'outputs':target, 'inputs':current_word_id}
                            yield output_dict




    def keep_word(self, word):
        z = self.frequency[word] / self.tot_occurrences
        p_keep = np.sqrt(z / 10e-3) + 1
        p_keep *= 10e-3 / z 
        return np.random.rand() < p_keep



