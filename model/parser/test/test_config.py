import os
import sys
sys.path.append('.')
sys.path.append('..')

from infra.config import Arguments

sys.argv = """--cuda --mode FastLSTM --num_epochs 1000 --batch_size 32 --hidden_size 512 --num_layers 3 \
 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --eps 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 \
 --p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --use_pos --use_char \
 --objective cross_entropy --decode mst \
 --word_embedding sskip --word_path "/home/zfhu/data/sskip.eng.100.gz" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train_path "/home/zfhu/data/ptb/train.conll" \
 --dev_path "/home/zfhu/data/ptb/dev.conll" \
 --test_path "/home/zfhu/data/ptb/test.conll" \
 --save_to "models/parsing/graph/"
""".split()
args = Arguments()
print(args.__dict__)