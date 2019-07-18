
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import WordpieceTokenizer

import logging
LOGGER = logging.getLogger()

class TedDataset(Dataset):
    def __init__(self,en_tokenizer,de_tokenizer,max_seq_len):
        with open('./dataset/de-en/train.en',encoding='utf-8') as f:
            en = f.readlines()
        with open('./dataset/de-en/train.de',encoding='utf-8') as f:
            de = f.readlines()
        assert(len(en)==len(de))
        self.len = len(en)
        self.en_data = en
        self.de_data = de
        self.en_tokenizer = en_tokenizer
        self.de_tokenizer = de_tokenizer
        self.max_seq_len = max_seq_len
    
    def __getitem__(self,index):
        en_data = self.en_data[index]
        de_data = self.de_data[index]
        # print(en_data)
        # print(de_data)
        return torch.tensor(self.en_tokenizer.transform(en_data)), torch.tensor(self.de_tokenizer.transform(de_data))
    
    def __len__(self):
        return self.len
    
    def collate_fn(self,data):
        """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
        We should build a custom collate_fn rather than using default collate_fn,
        because merging sequences (including padding) is not supported in default.
        Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
        Args:
            data: list of tuple (src_seq, trg_seq).
                - src_seq: torch tensor of shape (?); variable length.
                - trg_seq: torch tensor of shape (?); variable length.
        Returns:
            src_seqs: torch tensor of shape (batch_size, padded_length).
            src_lengths: list of length (batch_size); valid length for each padded source sequence.
            trg_seqs: torch tensor of shape (batch_size, padded_length).
            trg_lengths: list of length (batch_size); valid length for each padded target sequence.
        """
        def merge(sequences):
            padded_seqs = torch.zeros(len(sequences),self.max_seq_len).long()
            # print("padded_seqs.size()=",padded_seqs.size())
            for i, seq in enumerate(sequences):
                # print("min(self.max_seq_len,len(seq))=",min(self.max_seq_len,len(seq)))
                padded_seqs[i][:min(self.max_seq_len,len(seq))] = seq[:min(self.max_seq_len,len(seq))]
            return padded_seqs

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        src_seqs, trg_seqs = zip(*data)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        src_seqs = merge(src_seqs)
        trg_seqs = merge(trg_seqs)

        return src_seqs, trg_seqs

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)
            
def test():
    init_logging()
    en_tokenizer = WordpieceTokenizer('en').load_model()
    de_tokenizer = WordpieceTokenizer('de').load_model()
    
    # en_tokenizer.load_model()
    dataset = TedDataset(en_tokenizer=en_tokenizer, de_tokenizer=de_tokenizer, max_seq_len=50)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=3,
                              shuffle=False,
                              collate_fn=dataset.collate_fn)
    for i, data in enumerate(train_loader):
        inputs, outputs = data
        print("inputs.size()=",outputs.size())
        LOGGER.info("inputs.size()=".format(inputs.size()))
        LOGGER.info(inputs)
        print("outputs.size()=",outputs.size())
        LOGGER.info("outputs.size()=".format(outputs.size()))
        LOGGER.info(outputs)
        break
if __name__ == "__main__":
    test()