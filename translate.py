import torch
from model import TransformerModel
from tokenizer import WordpieceTokenizer
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import Dataset, DataLoader
from dataset import TedDataset
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load tokenizer
de_tokenizer = WordpieceTokenizer('de').load_model()
en_tokenizer = WordpieceTokenizer('en').load_model()

model = TransformerModel(d_model=512, 
                        num_heads=8, 
                        num_encoders=6, 
                        num_decoders=6,
                        in_vocab_size=len(de_tokenizer), 
                        out_vocab_size=len(en_tokenizer)).to(device)

model.load_state_dict(torch.load("./outputs/model-epoch10.pt"))
model.eval()

def translate(inputs):
    input_len = len(inputs)
    inputs = torch.tensor([de_tokenizer.transform(input,max_length=50) for input in inputs]).cuda()
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
            print("len(i)=",len(i))
            print("no eos token found")
        cleanoutput.append(i)
    outputs = cleanoutput
    return en_tokenizer.decode(outputs)

def get_blue_score():
    with open('./dataset/de-en/test.de'.format(type),encoding='utf-8') as f:
        inputs = f.readlines()
    with open('./dataset/de-en/test.en'.format(type),encoding='utf-8') as f:
        outputs = f.readlines()

    bleu_score = 0.
    batch_size = 1
    for minibatch in [outputs[i:i + batch_size] for i in range(0, len(outputs), batch_size)]:
        predictions = translate(minibatch)
        for i in range(len(minibatch)):
            bleu_score += sentence_bleu([minibatch[i].strip().split()], predictions[i].strip().split())
    bleu_score /= len(outputs)
    return bleu_score

bleu_score = get_blue_score()
print("bleu_score=",bleu_score)