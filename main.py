import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import WordpieceTokenizer
from dataset import TedDataset

# Load tokenizer
en_tokenizer = WordpieceTokenizer('en').load_model()
de_tokenizer = WordpieceTokenizer('de').load_model()
    
# Load dataset
dataset = TedDataset(en_tokenizer=en_tokenizer, de_tokenizer=de_tokenizer)
train_loader = DataLoader(dataset=dataset,
                            batch_size=3,
                            shuffle=False,
                            collate_fn=dataset.collate_fn)

for i, data in enumerate(train_loader):
    inputs, outputs = data
    # print("inputs.size()=",inputs.size())
    # print(inputs)
    # print("outputs.size()=",outputs.size())
    # print(outputs)
    break