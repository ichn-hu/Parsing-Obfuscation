from os.path import join

from iomodule.maxio import create_alphabets, read_data_to_variable

ptb_root = '/home/zfhu/data/ptb'
train_path = join(ptb_root, 'train.conll')
dev_path = join(ptb_root, 'dev.conll')
test_path = join(ptb_root, 'test.conll')
alphabets = create_alphabets('saved_alphabets', train_path, data_paths=[dev_path, test_path])

data_train = read_data_to_variable(train_path, *alphabets, use_gpu=True, symbolic_root=True)
data_dev = read_data_to_variable(dev_path, *alphabets, use_gpu=True, symbolic_root=True)
data_test = read_data_to_variable(test_path, *alphabets, use_gpu=True, symbolic_root=True)

