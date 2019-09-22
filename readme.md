# Transformer
Re-implementation of Attention Is All You Need (NIPS 2017)

## Create Conda Environment
```
$ conda env create -f env.yml
$ conda activate transformer
```

## Fit tokenizer
- It fits wordpiece tokenizer with 16000 subwords.
```
$ python tokenizer.py
```

## Train model
```
$ python main.py --mode train --model_name [model_name]
```

## Calculate BLEU score
- Download iwslt17 fr-en
```
$ sacrebleu -t iwslt17 -l fr-en --echo src > ./iwslt17-fr-en.src
```

- Translate evaluation set
```
$ python main.py --mode test --model_name [model_name]
```

- Score the decoder
```
$ cat output.detok.txt | sacrebleu -t iwslt17 -l fr-en
```