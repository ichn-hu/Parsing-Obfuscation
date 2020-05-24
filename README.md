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

We used the PTB dataset for testing on constituency parsing, which can be obtained [here](https://github.com/jhcross/span-parser/tree/master/data).

### Pretrained model

For the dependency parser, we used the BiAffine parser obtained using [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2). For the training of the obfuscator, while using the pretrained parser provided by [AllenNLP](https://demo.allennlp.org/dependency-parsing).

The pretrained word embedding can be obtained [here](https://github.com/XuezheMax/NeuroNLP2/issues/35).

### Training

See script `run.sh`, and commands in `utils/`.
