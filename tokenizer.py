import argparse
import sentencepiece as spm
import logging

LOGGER = logging.getLogger()

def get_args():
    parser = argparse.ArgumentParser()
    
    # Run settings
    parser.add_argument('--vocab_size', 
                        default=16000, type=int)
    parser.add_argument('--datapath', 
                        default='./datasets/iwslt17.fr.en')
    parser.add_argument('--src_path', 
                        default='./datasets/iwslt17.fr.en/train.fr-en.en')
    parser.add_argument('--dest_path', 
                        default='./datasets/iwslt17.fr.en/train.fr-en.fr')
    parser.add_argument('--langpair',
                        default='fr-en')

    args = parser.parse_args()
    return args

class WordpieceTokenizer(object):
    def __init__(self, datapath, vocab_size=0, l=0, alpha=0, n=0):
        logging.info("vocab_size={}".format(vocab_size))
        self.templates = '--input={} --model_prefix={} --vocab_size={} --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1'
        self.vocab_size = vocab_size
        self.spm_path = "{}/sp".format(datapath)

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
        cmd = self.templates.format(input_file, self.spm_path, self.vocab_size, 0)
        spm.SentencePieceTrainer.Train(cmd)
        
    def load_model(self):
        file = self.spm_path + ".model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(file)
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
        return len(self.sp)

def main():
    """
    Train SentencePiece Model
    """
    args = get_args()
    tokenizer = WordpieceTokenizer(datapath=args.datapath,
                                   vocab_size=args.vocab_size)
    tokenizer.fit(",".join([args.src_path,args.dest_path]))
    
if __name__ == "__main__":
    main()