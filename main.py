import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizer import WordpieceTokenizer
from dataset import TedDataset
from model import TransformerModel
import argparse
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
    
# Run settings
parser.add_argument('--max_seq_len', 
                    default=30, type=int)
parser.add_argument('--learning_rate', 
                    default=0.001, type=float)
parser.add_argument('--epoch', 
                    default=10, type=int)
parser.add_argument('--batch', 
                    default=64, type=int)

args = parser.parse_args()
                    
# Load tokenizer
de_tokenizer = WordpieceTokenizer('de').load_model()
en_tokenizer = WordpieceTokenizer('en').load_model()
    
# Load dataset
dataset = TedDataset(en_tokenizer=en_tokenizer,de_tokenizer=de_tokenizer,max_seq_len=args.max_seq_len)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)
val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

model = TransformerModel(d_model=512, 
                        num_heads=8, 
                        num_encoders=6, 
                        num_decoders=6,
                        seq_len=args.max_seq_len, 
                        in_vocab_size=len(en_tokenizer), 
                        out_vocab_size=len(de_tokenizer)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
# optimizer = optim.Adam(params, lr=self.learning_rate)

for epoch in range(args.epoch):
    train_loss = 0
    val_loss = 0
    # train
    model.train()
    for i, data in tqdm(enumerate(train_loader),total=len(train_loader)):
        optimizer.zero_grad()
        inputs, outputs = data
        output_probabilities = model(inputs,outputs)
        loss = criterion(output_probabilities.view(-1,len(de_tokenizer)), outputs.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*len(inputs[0])
        # break
    avg_train_loss = train_loss/train_size
    # avg_train_loss = train_loss/64

    # val
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader),total=len(val_loader)):
            inputs, outputs = data
            output_probabilities = model(inputs,outputs)
            loss = criterion(output_probabilities.view(-1,len(de_tokenizer)), outputs.view(-1))
            val_loss += loss.item()*len(inputs[0])
            # break
        
    avg_val_loss = val_loss/val_size
    # avg_val_loss = val_loss/64
    
    # result
    print("Epoch {}/{}, Train_Loss: {:.3f}, Val_Loss: {:.3f}".format(epoch+1,args.epoch, avg_train_loss, avg_val_loss))