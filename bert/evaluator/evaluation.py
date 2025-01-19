from torch.utils.data import DataLoader
import torch as t, tqdm
import torch.nn as nn
from ..model.language_model import BERTLM, BERT
from ..dataset import WordVocab, BERTDataset

class BERTEvaluator:
    
    def __init__(self, vocab: WordVocab, eval_dataset: BERTDataset, batch_size: int, checkpoint_model_path: str, seq_len: int, path_to_save_output: str):
        assert t.cuda.is_available()
        
        self.seq_len = seq_len
        
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        
        print("Loading model checkpoint...")
        self.model = t.load(checkpoint_model_path, map_location=t.device('cuda'))
        
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        self.path_to_save_output = path_to_save_output
        self.vocab = vocab
        self.criterion = nn.NLLLoss(ignore_index=0)
        
    
    def evaluate(self):
        
        self.model.eval()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.eval_dataloader),
                            desc="Evaluating",
                            total=len(self.eval_dataloader),
                            bar_format="{l_bar}{r_bar}")
        
        avg_loss = 0.0
        
        entire_sequence = t.tensor([], device=t.device("cpu"))
        
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the masked_lm model
            with t.no_grad():
                mask_lm_output: t.Tensor = self.model.forward(data["bert_input"])
            
            assert  mask_lm_output.shape[0] == self.batch_size and \
                    mask_lm_output.shape[1] == self.seq_len and \
                    mask_lm_output.shape[2] == len(self.vocab)

            entire_sequence = t.cat((entire_sequence, mask_lm_output.view(self.batch_size * self.seq_len, len(self.vocab)).cpu()))

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = mask_loss

            avg_loss += loss.item()
        
        # Take the token indices predicted by the model
        entire_sequence = entire_sequence.max(1).indices
        
        # Take the tokens (along with specials)
        entire_sequence = t.tensor([self.vocab.itos[idx] for idx in entire_sequence])
        mask_special_token = t.isin(entire_sequence, t.tensor(self.vocab.get_special_tokens()))
        
        # Remove special tokens and decrement all indices to bring them back to VQ-VAE token indices
        entire_sequence = entire_sequence[~mask_special_token] - len(self.vocab.get_special_tokens())
        
        original_len_of_input_sequence = self.eval_dataset.filenames_with_len_seq[0]['len']
        assert entire_sequence.shape[0] == original_len_of_input_sequence, f"Original length was [{original_len_of_input_sequence}], reconstructed sequence length is [{entire_sequence.shape[0]}]"
            
        avg_loss = avg_loss / len(data_iter)
        print(f"Average Loss: {avg_loss}")
        print(f"Saving output at '{self.path_to_save_output}'")
        t.save(entire_sequence, self.path_to_save_output)
        
        
        
        
        