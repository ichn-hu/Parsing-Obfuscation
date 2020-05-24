#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1

PROJ_PATH="/mnt/e/Projects"
ROOT_PATH="${PROJ_PATH}/Homomorphic-Obfuscation"
DATA_PATH="${PROJ_PATH}/data"
WORK_PATH="${PROJ_PATH}/work"
CACHE_PATH="${PROJ_PATH}/memfs"
MAIN_PATH="${ROOT_PATH}/main.py"

PYTHONPATH="${ROOT_PATH}/model/parser:{$PYTHONPATH}" CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $MAIN_PATH --exp_name FeedForwardObfuscator --mode FastLSTM \
 --model ffobf \
 --num_epochs 1000 --batch_size 32 --hidden_size 512 --num_layers 3 \
 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --eps 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 \
 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --use_word --use_pos --use_char \
 --objective cross_entropy --decode mst \
 --word_embedding sskip --word_path "${DATA_PATH}/sskip.eng.100.gz" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train_path "${DATA_PATH}/ptb/train.conll" \
 --dev_path "${DATA_PATH}/ptb/dev.conll" \
 --test_path "${DATA_PATH}/ptb/test.conll" \
 --save_to "${WORK_PATH}/save"
