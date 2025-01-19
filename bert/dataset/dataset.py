from torch.utils.data import Dataset
import random, os
import torch as t
from .vocab import WordVocab

class BERTDataset(Dataset):
    def __init__(self, path_to_data, vocab, seq_len, elements_to_mask=32, seed=None, evaluation=False, max_dataset_elements=None):
        '''
            :param seq_len: model input sequence
        '''
        print("Creating the dataset")
        if not evaluation:
            random.seed(seed)
            
        self.elements_to_mask = elements_to_mask # for generative task
        self.evaluation = evaluation
        
        self.vocab: WordVocab = vocab
        self.seq_len = seq_len - 2 # since we are adding SOS and EOS tokens to the input sequence
        self._load_filenames(path_to_data)
        self._load_sequence(max_dataset_elements)
        print(f"Dataset successfully created!")
        
    def _load_filenames(self, path_to_data):
        # We are going to build a list of filenames along with the length of the embedding sequence 
        # for example [['path/to/file_1', N_1], ['path/to/file_2', N_2], ...]
        self.filenames_with_len_seq = []
        
        for filename in os.listdir(path_to_data):
            file_path = os.path.join(path_to_data, filename)
            if os.path.isfile(file_path) and file_path.endswith(".pt"):
                self.filenames_with_len_seq.append({'file_path': file_path, 'len': None})
                
        assert not self.evaluation or len(self.filenames_with_len_seq) == 1, f"During the evaluation you must pass a directory containing only one file to build the dataset!"
                
    def _load_sequence(self, max_dataset_elements: int):
        sequences = t.tensor([])
        
        for l in self.filenames_with_len_seq:
            filename = l['file_path']
            print(f"Loading {filename}...")
            
            file_embedding_sequence: t.Tensor = t.load(filename)
            
            # Set the length of just loaded sequence
            l['len'] = file_embedding_sequence.shape[0]
            
            # Calculate the padding to add at the end
            file_embedding_sequence = file_embedding_sequence + len(self.vocab.get_special_tokens())
            padding = self.seq_len - file_embedding_sequence.shape[0] % self.seq_len
            padding = t.zeros(padding, dtype=file_embedding_sequence.dtype) + self.vocab.pad_index
            file_embedding_sequence = t.cat((file_embedding_sequence, padding))
            
            assert file_embedding_sequence.shape[0] % self.seq_len == 0, f"Got [{file_embedding_sequence.shape[0]}] and seq_len of [{self.seq_len}] "
            print(f"Loaded embeddings at '{filename}' of length [{file_embedding_sequence.shape[0]}] with padding of [{padding.shape[0]}]")
            sequences = t.cat((sequences, file_embedding_sequence))
            
        assert sequences.shape[0] % self.seq_len == 0
        sequences = sequences.view(sequences.shape[0]//self.seq_len, self.seq_len)
        sequences = sequences[:min(sequences.shape[0], max_dataset_elements), :] if max_dataset_elements else sequences
        print(f"Created a dataset of {sequences.shape[0]} elements")
        self.sequences = sequences.tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        input_sequence = self.sequences[item]
        
        if self.evaluation:
            masked_sequence, label = self.get_masked_sequence_in_the_middle(input_sequence)
        else:
            masked_sequence, label = self.random_embedding(input_sequence)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        bert_input = [self.vocab.sos_index] + masked_sequence + [self.vocab.eos_index]
        bert_label = [self.vocab.pad_index] + label + [self.vocab.pad_index]

        #padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        #bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label}

        return {key: t.tensor(value) for key, value in output.items()}
    
    def get_masked_sequence_in_the_middle(self, sequence):
        ''' For "generative" task.
            It returns the indices of each token in the itos. The returned sequence is masked in the middle
        '''
        elements_to_mask = self.elements_to_mask
        assert elements_to_mask % 2 == 0 and len(sequence) % 2 == 0
        
        output_labels = []
        start_masking = (len(sequence) - elements_to_mask) // 2
        end_masking = start_masking + elements_to_mask
        
        for i, token in enumerate(sequence):
            if i >= start_masking and i<=end_masking and token not in self.vocab.get_special_tokens():
                # mask the central tokens
                sequence[i] = self.vocab.mask_index
            else:
                # leave the other tokens unmasked
                sequence[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
            
            output_labels.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        
        return sequence, output_labels
            

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

        return sequence, output_label
