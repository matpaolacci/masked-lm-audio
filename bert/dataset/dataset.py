from torch.utils.data import Dataset
import random, os
import torch as t
from .vocab import WordVocab

class BERTDataset(Dataset):
    def __init__(self, path_to_data, vocab, seq_len):
        '''
            :param seq_len: model input sequence
        '''
        self.vocab: WordVocab = vocab
        self.seq_len = seq_len - 2 # since we are adding SOS and EOS tokens to the input sequence
        self._load_filenames(path_to_data)
        self._load_sequence()
        print(f"Dataset created with {self.__len__()} elements")
        
    def _load_filenames(self, path_to_data):
        self.filenames = []
        for filename in os.listdir(path_to_data):
            file_path = os.path.join(path_to_data, filename)
            if os.path.isfile(file_path):
                self.filenames.append(file_path)
                
    def _load_sequence(self):
        '''TODO: it takes only the first file'''
        file_embedding_sequence: t.Tensor = t.load(self.filenames[0])
        embds_to_remove = file_embedding_sequence.shape[0] % self.seq_len
        file_embedding_sequence = file_embedding_sequence[:file_embedding_sequence.shape[0]-embds_to_remove]
        self.batches: t.Tensor = file_embedding_sequence.view(file_embedding_sequence.shape[0]//self.seq_len, self.seq_len).tolist()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        input_sequence = self.batches[item]
        
        t1_random, t1_label = input_sequence, self.random_embedding(input_sequence)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]

        bert_input = t1
        bert_label = t1_label

        #padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        #bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label}

        return {key: t.tensor(value) for key, value in output.items()}

    def random_embedding(self, sequence):
        '''
            It takes a sequence of embeddings (indices) and replaces 15% of them by following the masking procedure in BERT.
            :return : the output_label
            
            :param sequence: list of the audio embedding indexes
        '''
        output_label = [] # it will filled with the true label

        for i, token in enumerate(sequence):
            prob = random.random()
            if prob < 0.15:
                # The probability is rescaled by dividing it by 0.15 (bringing it back to a range between 0 and 1)
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    sequence[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    sequence[i] = random.randrange(len(self.vocab)) 

                # 10% randomly change token to current token
                else:
                    # stoi works like this: given a word, it returns its index as found in itos
                    sequence[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            # Leave 85% of the tokens as they are, they will not concur in the loss calculation
            #   We label them with index=0
            else:
                sequence[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                # Indexes 0 will be ignored when we will calculate the loss
                output_label.append(0)

        return output_label
