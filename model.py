import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging

LOGGER = logging.getLogger()

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

class Self_Attention(nn.Module):
    def __init__(self,d_model = 512,
                num_heads=8,
                is_mask=False):
        super(Self_Attention, self).__init__()
        self.d_model= d_model
        self.num_heads = num_heads
        self.is_mask = is_mask

        self.d_q = int(d_model/num_heads)
        self.d_k = int(d_model/num_heads)
        self.d_v = int(d_model/num_heads)

        self.linear_q = nn.Linear(self.d_model, self.d_q)
        self.linear_k = nn.Linear(self.d_model, self.d_k)
        self.linear_v = nn.Linear(self.d_model, self.d_v)

        self.linear = nn.Linear(self.num_heads*self.d_v,self.d_model)

    def forward(self, Q, K, V):
        Q = self.linear_q(Q)
        K = self.linear_k(K)
        V = self.linear_v(V)
        x = self.multi_head_attention(Q,K,V)
        x = self.linear(x)

        return x

    def single_attention(self,Q,K,V):
        """
        scaled dot-production attention
        Q: Queries
        K: Keys
        V: Values
        """
        # 1. batch matrix multiplication of Q and K
        x = torch.bmm(Q,K.transpose(1,2))
        # LOGGER.info("after matmul\n{}".format(x))
        
        # 2. scale
        x = x/math.sqrt(self.d_k)
        # LOGGER.info("after scale\n{}".format(x)) 

        # 3. mask (option)
        if self.is_mask:
            seq_len = Q.size()[-2]
            x = self.mask(x,seq_len)
            # LOGGER.info("after masking\n{}".format(x))

        # 4. softmax
        x = F.softmax(x,dim=1)
        # LOGGER.info("after softmax\n{}".format(x)) 

        # 5. batch matrix multiplication with V
        x = torch.bmm(x,V)
        # LOGGER.info("after matmul\n{}".format(x))

        return x

    def multi_head_attention(self,Q,K,V):
        """
        multi head attention
        """
        output = []
        for head in range(self.num_heads):
            output.append(self.single_attention(Q,K,V))
        x = torch.cat(output,dim=2)
        return x

    def mask(self,input, seq_len):
        masked_input = torch.triu(torch.ones(seq_len,seq_len),diagonal=1)*(-1.0e9)
        if torch.cuda.is_available():
            masked_input = masked_input.cuda()
        masked_input = masked_input + input
        return masked_input

class Encoder(nn.Module):
    def __init__(self, seq_len = 4, d_model = 512, num_heads = 8):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_heads = 8
        self.self_attention = Self_Attention(self.d_model, self.num_heads)
        self.ffnn = nn.Sequential(
            nn.Linear(self.d_model,2048),
            nn.ReLU(),
            nn.Linear(2048, self.d_model)
        )
        self.norm = nn.LayerNorm((seq_len,d_model))
        self.dropout = nn.Dropout(0.1)

    def forward(self,input):
        # 1. Multi head attention
        x = Q = K = V = input
        x = self.norm (x + self.dropout(self.self_attention(Q,K,V)))

        # 2. Feed Forward
        x = self.norm (x + self.dropout(self.ffnn(x)))

        return x

class Decoder(nn.Module):
    def __init__(self, seq_len = 4, d_model = 512, num_heads = 8):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_heads = 8
        self.self_attention = Self_Attention(self.d_model,self.num_heads, is_mask=False)
        self.masked_self_attention = Self_Attention(self.d_model, self.num_heads, is_mask=True)
        self.ffnn = nn.Sequential(
            nn.Linear(self.d_model,2048),
            nn.ReLU(),
            nn.Linear(2048, self.d_model)
        )
        self.norm = nn.LayerNorm((seq_len,d_model))
        self.dropout = nn.Dropout(0.1)

    def forward(self,input, output):
        # 1. Masked multi head attention 
        x = Q = K = V = output
        x = self.norm (x + self.dropout(self.masked_self_attention(Q,K,V)))
        
        # 2. Multi head attention
        Q = x
        K = V = input
        x = self.norm (x + self.dropout(self.self_attention(Q,K,V)))

        # 3. Feed Forward
        x = self.norm (x + self.dropout(self.ffnn(x)))

        return x

class WordEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 output_dim=512):
        super(WordEmbedding, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = output_dim,
            padding_idx = 0)
            
    def forward(self,inputs):
        x = self.embed(inputs)
        
        seq_len = x.size()[1]
        d_model = x.size()[2]
        pos_encoding = self.position_encoding(seq_len,d_model)
        x += pos_encoding  
        
        return x
    
    def position_encoding(self,seq_len, d_model):
        """
        same dimensiton d_model as the embeddings
        """
        positions = np.arange(seq_len)[:, np.newaxis]
        dimentions = np.arange(d_model)[np.newaxis,:]
        angles = positions/ np.power(10000, 2*(dimentions//2)/d_model)

        pos_encoding = np.zeros(angles.shape)
        pos_encoding[:,0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:,1::2] = np.cos(angles[:, 1::2])
        
        pos_encoding = torch.FloatTensor(pos_encoding)
        if torch.cuda.is_available():
            pos_encoding = pos_encoding.cuda()
        return pos_encoding

class TransformerModel(nn.Module):
    def __init__(self, d_model= 512, 
                        num_heads=8,
                        num_encoders=6,
                        num_decoders=6,
                        seq_len = 20,
                        in_vocab_size=32000,
                        out_vocab_size=32000):
        super(TransformerModel,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size

        self.in_word_embed = WordEmbedding(vocab_size=self.in_vocab_size)
        self.out_word_embed = WordEmbedding(vocab_size=self.out_vocab_size)
        self.encoders = nn.ModuleList([Encoder(self.seq_len, self.d_model ,self.num_heads) for i in range(self.num_encoders)])
        self.decoders = nn.ModuleList([Decoder(self.seq_len, self.d_model ,self.num_heads) for i in range(self.num_decoders)])

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.d_model,self.out_vocab_size)
        # self.softmax = nn.Softmax(dim=-1)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,input,output):
        input = self.in_word_embed(input)
        input = self.dropout(input)
        LOGGER.info("input size={}".format(input.size()))

        output = self.out_word_embed(output)
        output = self.dropout(output)
        LOGGER.info("output size={}".format(output.size()))

        encoded_input = input
        for encoder in self.encoders:
            encoded_input = encoder(encoded_input)
        LOGGER.info("encoded_input size={}".format(encoded_input.size()))

        decoded_output = output
        for decoder in self.decoders:
            decoded_output = decoder(encoded_input, decoded_output)
        LOGGER.info("decoded_output size={}".format(decoded_output.size()))

        x = self.linear(decoded_output)
        output_probabilities = self.softmax(x)
        LOGGER.info("output probabilities size={}".format(output_probabilities.size()))
        
        return output_probabilities
        
def test():
    init_logging()
    inputs = torch.LongTensor([[1,2,3,4]])   
    outputs = torch.LongTensor([[1,2,3,4]])    

    model = TransformerModel(d_model=512, num_heads=8, num_encoders=6, num_decoders=6,seq_len=4, in_vocab_size=20, out_vocab_size=20)
    output_probabilities = model(inputs,outputs)

if __name__ == "__main__":
    test()
