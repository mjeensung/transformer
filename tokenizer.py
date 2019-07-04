import sentencepiece as spm

class WordpieceTokenizer(object):
    def __init__(self,prefix):
        self.templates = '--input={} --model_prefix={} --vocab_size={}'
        self.vocab_size = 5000
        self.prefix = "./dataset/"+prefix
        
    def transform(self,sentence):
        x = self.sp.EncodeAsPieces(sentence)
        print(x)
        x = self.sp.EncodeAsIds(sentence)
        return x
    
    def fit(self,input_file):
        cmd = self.templates.format(input_file, self.prefix, self.vocab_size)
        spm.SentencePieceTrainer.Train(cmd)
        
    def load_model(self):
        file = self.prefix+".model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.prefix+".model")
        print("load_model {}".format(file))
        return self

def test():
    """
    Generate tokenizer
    """
    en_tokenizer = WordpieceTokenizer('en')
    en_tokenizer.fit("./dataset/de-en/train.en")
    de_tokenizer = WordpieceTokenizer('de')
    de_tokenizer.fit("./dataset/de-en/train.de")
    
    # en_tokenizer.load_model()
    # en_tokenizer.transform(["But it does not need to have to be that way.","Wernher von Braun, in the aftermath of World War II concluded, quote: science and religion are not antagonists."])
    
if __name__ == "__main__":
    test()