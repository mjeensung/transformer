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
import logging
import random
import numpy as np

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

def get_args():
    parser = argparse.ArgumentParser()
    
    # Run settings
    parser.add_argument('--max_seq_len', 
                        default=50, type=int)
    parser.add_argument('--learning_rate', 
                        default=0.0001, type=float)
    parser.add_argument('--epoch', 
                        default=200, type=int)
    parser.add_argument('--batch', 
                        default=64, type=int)
    parser.add_argument('--seed', 
                        default=0, type=int)                    
    parser.add_argument('--datapath', 
                        default='iwslt17')
    parser.add_argument('--langpair', 
                        default='fr-en')
    parser.add_argument('--model_name', 
                        default='model')
    parser.add_argument('--mode', 
                        default='train')
    
                                            
    # tokenization
    parser.add_argument('--l', 
                        default=0, type=int)
    parser.add_argument('--alpha', 
                        default=0, type=float)
    
    args = parser.parse_args()

    logger.info(f"device: {device}, n_gpu: {n_gpu}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    return args

def main(args):
    if args.mode == 'train':
        # Load tokenizer
        srctokenizer = WordpieceTokenizer(args.datapath,args.langpair,args.l, args.alpha).load_model()
        trgtokenizer = WordpieceTokenizer(args.datapath,args.langpair).load_model()
    
        # Load dataset
        train_dataset = TedDataset(type="train",srctokenizer=srctokenizer,trgtokenizer=trgtokenizer,
                                    max_seq_len=args.max_seq_len,datapath=args.datapath,langpair=args.langpair)
        val_dataset = TedDataset(type="valid",srctokenizer=srctokenizer,trgtokenizer=trgtokenizer,
                                    max_seq_len=args.max_seq_len,datapath=args.datapath,langpair=args.langpair)

        train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=args.batch,
                                    shuffle=True,
                                    collate_fn=train_dataset.collate_fn)
        val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=args.batch,
                                    shuffle=False,
                                    collate_fn=val_dataset.collate_fn)

        model = TransformerModel(d_model=512, 
                                num_heads=8, 
                                num_encoders=6, 
                                num_decoders=6,
                                # seq_len=args.max_seq_len, 
                                in_vocab_size=len(srctokenizer), 
                                out_vocab_size=len(trgtokenizer)).to(device)
        criterion = nn.NLLLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,betas=(0.9, 0.98), eps=1e-09)

        best_loss = float("inf")
        cnt = 0
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
                loss = criterion(output_probabilities.view(-1,len(trgtokenizer)), targets.view(-1))
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
            # print("train!outputs=",outputs.tolist())
            # print("train!predict=",torch.argmax(output_probabilities,dim=-1).tolist())
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
                    loss = criterion(output_probabilities.view(-1,len(trgtokenizer)), targets.view(-1))
                    val_loss += loss.item()*len(outputs)
                    val_total += len(outputs)
                    # break
                val_loss /= val_total
                print("val!outputs=",trgtokenizer.decode(outputs.tolist()))
                print("val!predict=",trgtokenizer.decode(torch.argmax(output_probabilities,dim=-1).tolist()))
            # result
            print("Epoch {}/{}, Train_Loss: {:.3f}, Val_Loss: {:.3f}".format(epoch+1,args.epoch, train_loss, val_loss))
            if best_loss > val_loss:
                print("model saved!")
                best_loss = val_loss
                torch.save(model.state_dict(), "outputs/{}.pt".format(args.model_name))
                cnt = 0 
            else:
                cnt += 1

            if cnt>3:
                break
    elif args.mode == 'test':
        # Load tokenizer
        srctokenizer = WordpieceTokenizer(args.datapath,args.langpair).load_model()
        trgtokenizer = WordpieceTokenizer(args.datapath,args.langpair).load_model()
    
        model = TransformerModel(d_model=512, 
                                num_heads=8, 
                                num_encoders=6, 
                                num_decoders=6,
                                in_vocab_size=len(srctokenizer), 
                                out_vocab_size=len(trgtokenizer)).to(device)

        model.load_state_dict(torch.load("./outputs/{}.pt".format(args.model_name)))
        model.eval()

        def translate(inputs):
            input_len = len(inputs)
            inputs = torch.tensor([srctokenizer.transform(input,max_length=50) for input in inputs]).cuda()
            outputs = torch.tensor([[2]]*input_len).cuda() #2 means sos token
            for i in range(50):
                prediction = model(inputs,outputs)
                prediction = torch.argmax(prediction,dim=-1)[:,-1] # get final token
                outputs = torch.cat((outputs,prediction.view(-1,1)),dim=-1)
            outputs = outputs.tolist()
            cleanoutput = []
            for i in outputs:
                try:
                    eos_idx = i.index(3) # 3 means eos token
                    i = i[:eos_idx]
                except:
                    pass
                    # print("len(i)=",len(i))
                    # print("no eos token found")
                cleanoutput.append(i)
            outputs = cleanoutput
            return trgtokenizer.decode(outputs)


        with open('./iwslt17-fr-en.src',mode='r',encoding='utf-8') as f:
            inputs = f.readlines()
        outputs = []        

        batch_size = 32
        for minibatch in tqdm([inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]):
            outputs += translate(minibatch)

        with open('./output.detok.txt'.format(args.model_name),mode='w',encoding='utf-8') as f:
            f.write('\n'.join(outputs) + '\n')

if __name__ == "__main__":
    main(get_args())