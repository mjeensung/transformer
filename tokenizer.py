import sentencepiece as spm
import logging

LOGGER = logging.getLogger()

class WordpieceTokenizer(object):
    def __init__(self,prefix,max_seq_len=50):
        self.templates = '--input={} --model_prefix={} --vocab_size={} --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1'
        self.vocab_size = 5000
        self.prefix = "./dataset/"+prefix
        self.max_seq_len = max_seq_len
        
    def transform(self,sentence, type='input'):
        if type=='input':
            self.sp.SetEncodeExtraOptions('eos')
        elif type=='output':
            self.sp.SetEncodeExtraOptions('bos:eos')
        x = self.sp.EncodeAsPieces(sentence)
        # LOGGER.info(x)
        x = self.sp.EncodeAsIds(sentence)

        # padding
        if len(x) < self.max_seq_len:
            padded_x = [0]*self.max_seq_len
            padded_x[:len(x)] = x
            x = padded_x
        
        return x
    
    def fit(self,input_file):
        cmd = self.templates.format(input_file, self.prefix, self.vocab_size, 0)
        spm.SentencePieceTrainer.Train(cmd)
        
    def load_model(self):
        file = self.prefix+".model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.prefix+".model")
        self.sp.SetEncodeExtraOptions('bos:eos')
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
    en_tokenizer = WordpieceTokenizer('en')
    # en_tokenizer.fit("./dataset/de-en/train.en")
    # de_tokenizer = WordpieceTokenizer('de')
    # de_tokenizer.fit("./dataset/de-en/train.de")
    
    en_tokenizer.load_model()

    a = en_tokenizer.transform("But it does not need to have to be that way.")
    print("a=",a)

    a = en_tokenizer.decode([a])
    print("a=",a)
if __name__ == "__main__":
    test()