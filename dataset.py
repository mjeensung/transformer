
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import WordpieceTokenizer
import logging
LOGGER = logging.getLogger()

class TedDataset(Dataset):
    def __init__(self, type, srctokenizer, trgtokenizer, max_seq_len, datapath, langpair):
        srclang = langpair.split("-")[0]
        trglang = langpair.split("-")[1]
        
        with open('./dataset/{}/{}.{}.{}'.format(datapath, type, langpair, srclang),encoding='utf-8') as f:
            src = f.readlines()
        with open('./dataset/{}/{}.{}.{}'.format(datapath, type, langpair, trglang),encoding='utf-8') as f:
            trg = f.readlines()
        assert(len(src)==len(trg))
        self.len = len(src)
        self.input = src
        self.output = trg
        self.srctokenizer = srctokenizer
        self.trgtokenizer = trgtokenizer
        self.max_seq_len = max_seq_len
    
    def __getitem__(self,index):
        input = self.input[index]
        output = self.output[index]
        return torch.tensor(self.srctokenizer.transform(input), dtype=torch.float64, requires_grad=True), torch.tensor(self.trgtokenizer.transform(output), dtype=torch.float64, requires_grad=True)
    
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
            # padded_seqs = torch.zeros(len(sequences),self.max_seq_len, requires_grad=True).long()
            padded_seqs = torch.zeros(len(sequences),self.max_seq_len, requires_grad=True).long()
            if torch.cuda.is_available():
                padded_seqs = padded_seqs.cuda()
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
    srctokenizer = WordpieceTokenizer('iwslt17','fr-en',l=64, alpha=0.1).load_model()
    trgtokenizer = WordpieceTokenizer('iwslt17','fr-en').load_model()

    # en_tokenizer.load_model()
    dataset = TedDataset(type='valid', srctokenizer=srctokenizer, trgtokenizer=trgtokenizer, 
                         max_seq_len=10, datapath='iwslt17',langpair='fr-en')
    train_loader = DataLoader(dataset=dataset,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=dataset.collate_fn)
    for epoch in range(10):
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