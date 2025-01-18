import argparse, json

from torch.utils.data import DataLoader
import torch as t

from .model import BERT, BERTLM
from .trainer import BERTTrainer
from .dataset import BERTDataset, WordVocab
from .evaluator import BERTEvaluator 

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--mode", required=True, type=str, help="set if you want train the model or do the inference")
    parser.add_argument("--path_to_saved_model", type=str, help="if you want make inference specify the saved model path")
    parser.add_argument("--output_dir", type=str, help="declare where you want save the output of the model")
    
    parser.add_argument("-c", "--path_to_train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--path_to_test_dataset", type=str, default=None, help="test set for evaluate train set")
    
    # evaluation
    parser.add_argument("--path_to_eval_dataset", type=str, default=None, required=False, help="eval dataset for eval bert")
    parser.add_argument("--path_to_save_output", type=str, default=None, required=False, help="path to save the output of bert")
    parser.add_argument("--elements_to_mask", type=int, default=None, required=False, help="number of elements to mask in the evaluating sequence")
    
    # train
    parser.add_argument("--schedule_optim_warmup_steps", type=int, default=10000, help="Controls gradual learning rate increase at the start")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("--max_dataset_elements", type=int, default=None, help="maximum elements in the datasets")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=None, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()
    
    if args.mode == "inference":
        inference(args)
    elif args.mode == "train":
        train(args)
    else:
        raise RuntimeError("--mode argument must be either 'infernce' or 'train'")


def train(args: argparse.Namespace):

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.path_to_train_dataset)
    train_dataset = BERTDataset(args.path_to_train_dataset, vocab, seq_len=args.seq_len, seed=args.random_seed, max_dataset_elements=args.max_dataset_elements)

    print("Loading Test Dataset", args.path_to_test_dataset)
    test_dataset = BERTDataset(args.path_to_test_dataset, vocab, seq_len=args.seq_len, seed=args.random_seed, max_dataset_elements=args.max_dataset_elements) \
        if args.path_to_test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), args.seq_len, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, checkpoint_path=args.output_path, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, warmup_steps=args.schedule_optim_warmup_steps)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)

        if test_data_loader is not None:
            trainer.test(epoch)
            
def inference(args: argparse.Namespace):
    
    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))
    
    print("Loading Evaluation Dataset", args.path_to_eval_dataset)
    eval_dataset = BERTDataset(
        args.path_to_eval_dataset, 
        vocab, 
        seq_len=args.seq_len,
        elements_to_mask=args.elements_to_mask,
        evaluation=True,
        seed=args.random_seed, 
        seq_len=args.seq_len,
        max_dataset_elements=args.max_dataset_elements
    )
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    print("Building BERT model")
    bert = BERT(len(vocab), args.seq_len, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
    
    bertEvaluator = BERTEvaluator(
        bert,
        vocab,
        seq_len=args.seq_len,
        checkpoint_model_path=args.path_to_model_checkpoint,
        path_to_save_output=args.path_to_save_output
    )
    
    bertEvaluator.evaluate(eval_dataloader)

if __name__ == '__main__':
    main()