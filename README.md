# Homomorphic-Obfuscation

This repository contains code for paper *Obfuscation for Privacy-preserving Syntactic Parsing* appeared in IWPT 2020.

## Introduction

The environment setup for the code includes

* python>=3.6
* torch>=1.0
* allennlp>=0.8.1

### Code hierarchy

* `config` includes configuration files for experiment setup
* `data` includes I/O facility and preprocessing functionality
* `net` has the model for attacker
* `model` has the model for the parser and the obfuscator

## How to run

### Dataset

We used the PTB dataset for testing on constituency parsing, which can be obtained [here](https://github.com/jhcross/span-parser/tree/master/data). The dataset will be used by allennlp's constituency parser on the output.

### Models

There are 3 components involved in the experiment, the parser as the service the user want to use, the obfuscator which obfuscates the input, and the attacker who wants to recover the original input. And for each components, there are different settings which results in different models.

We experimented with two kinds of parsers, one is a dependency parser and the other constituency parser, both pretrained. For the dependency parser, we used the BiAffine parser obtained using [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2) for the training of the obfuscator, and the pretrained dependency parser offered from [AllenNLP](https://demo.allennlp.org/dependency-parsing) for testing. For the constituency parser, we also used the pretrained model from [AllenNLP](https://demo.allennlp.org/constituency-parsing).

For the obfuscator/generator part, we have a random baseline model and a neural model, the code for both model can be found in `net/generator`.

As for the attacker, we experimented with a trained attacker and a BERT pretrained attacker. The code for the trained attacker is at `net/attacker.py` and the BERT attacker uses code from huggingface's `pytorch_pretrained_bert`, which is now at [huggingface/transformers](https://github.com/huggingface/transformers), and the code we used for the BERT attacker model is at `net/bert_attacker.py`.

Additionally, we used pretrained word embedding, which can be obtained [here](https://github.com/XuezheMax/NeuroNLP2/issues/35).

### Training

See script `run.sh`, and commands in `utils/`.

## Citation

> To be added
