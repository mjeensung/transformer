
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import WordpieceTokenizer

class TedDataset(Dataset):
    def __init__(self,en_tokenizer,de_tokenizer):
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
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        src_seqs, trg_seqs = zip(*data)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        src_seqs, src_lengths = merge(src_seqs)
        trg_seqs, trg_lengths = merge(trg_seqs)

        return src_seqs, trg_seqs
        
def test():
    en_tokenizer = WordpieceTokenizer('en').load_model()
    de_tokenizer = WordpieceTokenizer('de').load_model()
    
    # en_tokenizer.load_model()
    dataset = TedDataset(en_tokenizer=en_tokenizer, de_tokenizer=de_tokenizer)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=3,
                              shuffle=False,
                              collate_fn=dataset.collate_fn)
    for i, data in enumerate(train_loader):
        inputs, outputs = data
        print("inputs.size()=",inputs.size())
        print(inputs)
        print("outputs.size()=",outputs.size())
        print(outputs)
        break
if __name__ == "__main__":
    test()