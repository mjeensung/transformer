import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizer import WordpieceTokenizer
from dataset import TedDataset
from model import TransformerModel
import argparse
from tqdm import tqdm
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
    
# Run settings
parser.add_argument('--max_seq_len', 
                    default=50, type=int)
parser.add_argument('--learning_rate', 
                    default=0.0001, type=float)
parser.add_argument('--epoch', 
                    default=10, type=int)
parser.add_argument('--batch', 
                    default=64, type=int)

args = parser.parse_args()
                    
# Load tokenizer
de_tokenizer = WordpieceTokenizer('de').load_model()
en_tokenizer = WordpieceTokenizer('en').load_model()
    
# Load dataset
train_dataset = TedDataset(input_tokenizer=de_tokenizer,output_tokenizer=en_tokenizer,max_seq_len=args.max_seq_len,type="train")
val_dataset = TedDataset(input_tokenizer=de_tokenizer,output_tokenizer=en_tokenizer,max_seq_len=args.max_seq_len,type="val")

# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            collate_fn=train_dataset.collate_fn)
val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            collate_fn=val_dataset.collate_fn)

model = TransformerModel(d_model=512, 
                        num_heads=8, 
                        num_encoders=6, 
                        num_decoders=6,
                        # seq_len=args.max_seq_len, 
                        in_vocab_size=len(de_tokenizer), 
                        out_vocab_size=len(en_tokenizer)).to(device)
criterion = nn.NLLLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,betas=(0.9, 0.98), eps=1e-09)

for epoch in range(args.epoch):
    train_loss = 0
    val_loss = 0
    train_total = 0
    val_total = 0
    # train
    model.train()
    for i, data in tqdm(enumerate(train_loader),total=len(train_loader)):
        inputs, outputs = data
        targets = outputs
        bos_tokens = torch.ones(outputs.size()[0],1).long().cuda()*2 # 2 means sos token
        outputs = torch.cat((bos_tokens,outputs),dim=-1) # insert bos token in front
        outputs = outputs[:,:-1]
        output_probabilities = model(inputs,outputs)
        loss = criterion(output_probabilities.view(-1,len(en_tokenizer)), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_total += 1
        # if i % 10 == 0:
        #     print("train!outputs=",targets.tolist()[0])
        #     print("train!predict=",torch.argmax(output_probabilities,dim=-1).tolist()[0])
        # break
    train_loss /= train_total
    print("train!outputs=",outputs.tolist())
    print("train!predict=",torch.argmax(output_probabilities,dim=-1).tolist())
    # val
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader),total=len(val_loader)):
            inputs, outputs = data
            targets = outputs
            bos_tokens = torch.ones(outputs.size()[0],1).long().cuda()*2 # 2 means sos token
            outputs = torch.cat((bos_tokens,outputs),dim=-1) # insert bos token in front
            outputs = outputs[:,:-1]
            output_probabilities = model(inputs,outputs)
            loss = criterion(output_probabilities.view(-1,len(en_tokenizer)), targets.view(-1))
            val_loss += loss.item()*len(outputs)
            val_total += len(outputs)
            # break
        val_loss /= val_total
        print("val!outputs=",en_tokenizer.decode(outputs.tolist()))
        print("val!predict=",en_tokenizer.decode(torch.argmax(output_probabilities,dim=-1).tolist()))
    # result
    torch.save(model.state_dict(), "outputs/model-epoch{}.pt".format(epoch+1))
    print("Epoch {}/{}, Train_Loss: {:.3f}, Val_Loss: {:.3f}".format(epoch+1,args.epoch, train_loss, val_loss))