import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import WordpieceTokenizer
from dataset import TedDataset
from model import TransformerModel
import argparse

parser = argparse.ArgumentParser()
    
# Run settings
parser.add_argument('--max_seq_len', 
                    default=50, type=int)

args = parser.parse_args()
                    
# Load tokenizer
de_tokenizer = WordpieceTokenizer('de').load_model()
en_tokenizer = WordpieceTokenizer('en').load_model()
    
# Load dataset
dataset = TedDataset(en_tokenizer=en_tokenizer,de_tokenizer=de_tokenizer,max_seq_len=args.max_seq_len)
train_loader = DataLoader(dataset=dataset,
                            batch_size=3,
                            shuffle=False,
                            collate_fn=dataset.collate_fn)

for i, data in enumerate(train_loader):
    inputs, outputs = data
    # init_logging()
    # inputs = torch.LongTensor([[1,2,3,4]])   
    # outputs = torch.LongTensor([[1,2,3,4]])    

    model = TransformerModel(d_model=512, num_heads=8, num_encoders=6, num_decoders=6,seq_len=4, in_vocab_size=len(en_tokenizer), out_vocab_size=len(de_tokenizer))
    output_probabilities = model(inputs,outputs)
    print("output_probabilities.size()=",output_probabilities.size())
    break