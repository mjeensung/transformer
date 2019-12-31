# Transformer
Re-implementation of Attention Is All You Need (NIPS 2017)

## Requirements
- python=3.6.0
- pytorch=0.4.1
- sencencepiece=0.1.82

## Download datasets
Use IWSLT17 fr-en. The following script downloads datasets and preprocess them.
```
$ download.sh
```

## Sentence Piece Model
Use [SentencePiece](https://github.com/google/sentencepiece) tokenizer for subword-level segmentation in sentences.
```
$ python tokenizer.py --vocab_size 16000
```

## Train model
```
$ MODEL_NAME="model"
$ python main.py --mode train \
                 --datapath './datasets/iwslt17.fr.en'\
                 --langpair 'fr-en'\
                 --epoch 100\
                 --learning_rate 0.0001\
                 --max_seq_len 50\
                 --model_name ${MODEL_NAME}
```

## Evaluate
Use [SacreBLEU](https://github.com/mjpost/sacreBLEU) to evaluate the model based on BLEU score.
```
MODEL_NAME="model"
INPUT_NAME="./iwslt17-fr-en.in"
OUTPUT_NAME="./iwslt17-fr-en.out"
$ sacrebleu -t iwslt17 -l fr-en --echo src > ${INPUT_NAME}
$ python main.py --mode test \
                 --model_name ${MODEL_NAME} \
                 --eval_input ${INPUT_NAME} \
                 --eval_output ${OUTPUT_NAME}
$ cat ${OUTPUT_NAME} | sacrebleu -t iwslt17 -l fr-en
```