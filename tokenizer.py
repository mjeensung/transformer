import sentencepiece as spm
import logging

LOGGER = logging.getLogger()

class WordpieceTokenizer(object):
    def __init__(self, datapath, langpair, l=0, alpha=0, n=0):
        self.templates = '--input={} --model_prefix={} --vocab_size={} --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1'
        self.vocab_size = 16000
        self.langpair = "./dataset/{}/{}".format(datapath, langpair)
        # for subword regualarization
        self.l = l
        self.alpha = alpha
        self.n = n

        
    def transform(self,sentence, max_length=0):
        if self.l and self.alpha:
            x = self.sp.SampleEncodeAsIds(sentence, self.l, self.alpha)
        elif self.n:
            x = self.sp.NBestEncodeAsIds(sentence, self.n)
        else:
            x = self.sp.EncodeAsIds(sentence)
        if max_length>0:
            pad = [0]*max_length
            pad[:min(len(x),max_length)] = x[:min(len(x),max_length)]
            x = pad
        return x
    
    def fit(self, input_file):
        cmd = self.templates.format(input_file, self.langpair, self.vocab_size, 0)
        spm.SentencePieceTrainer.Train(cmd)
        
    def load_model(self):
        file = self.langpair+".model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.langpair+".model")
        self.sp.SetEncodeExtraOptions('eos')
        print("load_model {}".format(file))
        return self

    def decode(self,encoded_sentences):
        decoded_output = []
        for encoded_sentence in encoded_sentences:
            x = self.sp.DecodeIds(encoded_sentence)
            decoded_output.append(x)
        return decoded_output

    def __len__(self):
        return self.vocab_size

def test():
    """
    Generate tokenizer
    """
    tokenizer = WordpieceTokenizer('iwslt17','fr-en')
    tokenizer.fit("./dataset/iwslt17/train.fr-en.en,./dataset/iwslt17/train.fr-en.fr")
    
if __name__ == "__main__":
    test()