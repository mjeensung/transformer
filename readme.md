# Transformer
Re-implementation of Attention Is All You Need (NIPS 2017)

## Create Conda Environment
```
$ conda env create -f env.yml
$ conda activate transformer
```

## Fit tokenizer
- It fits wordpiece tokenizer with 5000 subwords.
```
$ python tokenizer.py
```

## Train model
```
$ python main.py
```

## Calculate BLEU score
```
$ python translate.py
```
