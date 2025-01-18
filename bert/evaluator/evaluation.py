from bert.model.bert import BERT
from torch.utils.data import DataLoader
import torch as t, tqdm
from bert.model.language_model import BERTLM
from dataset import WordVocab

class BERTEvaluator:
    
    def __init__(self, bert: BERT, vocab: WordVocab, checkpoint_model_path: str, seq_len: int, path_to_save_output: str):
        assert t.cuda.is_available()
        
        self.seq_len = seq_len
        
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.model = BERTLM(bert, len(vocab)).to(self.device)
        self.model.load_state_dict(
            t.load(checkpoint_model_path, map_location=t.device('cuda'))
        )
        
        self.path_to_save_output = path_to_save_output
        
        self.model.eval()
        
    
    def evaluate(self, eval_dataloader: DataLoader):

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(eval_dataloader),
                            desc="Evaluating",
                            total=len(eval_dataloader),
                            bar_format="{l_bar}{r_bar}")
        
        avg_loss = 0.0
        
        entire_sequence = t.tensor([])
        
        for _, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            mask_lm_output: t.Tensor = self.model.forward(data["bert_input"])
            
            assert mask_lm_output.shape[0] == eval_dataloader.batch_size and mask_lm_output.shape[1] == self.seq_len

            entire_sequence = t.cat((entire_sequence, mask_lm_output.view(mask_lm_output.shape[0] * mask_lm_output.shape[1])))

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = mask_loss

            avg_loss += loss.item()
            
        avg_loss = avg_loss / len(data_iter)
        print(f"Average Loss: {avg_loss}")
        print(f"Saving output at '{self.path_to_save_output}'")
        
        
        
        
        