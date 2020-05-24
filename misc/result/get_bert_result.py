import os

log_dir = '$ROOT/Project/parallel_log'

log_files = [
    '03.25_05:13-attack_with_bert-tagspec-ctx-use_pos-select:1.log',
    '03.25_05:10-attack_with_bert-tagspec-ctx-use_pos-select:2.log',
    '03.25_05:13-attack_with_bert-tagspec-ctx-use_pos-select:3.log',
    '03.25_05:14-attack_with_bert-tagspec-ctx-use_pos-select:4.log',
    '03.25_05:15-attack_with_bert-tagspec-ctx-use_pos-select:5.log',
    '03.25_05:13-attack_with_bert-tagspec-ctx-use_pos-select:6.log',
    '03.25_05:14-attack_with_bert-tagspec-ctx-use_pos-select:7.log',
    '03.25_05:14-attack_with_bert-tagspec-rnd-select:1.log',
    '03.25_05:14-attack_with_bert-tagspec-rnd-select:2.log',
    '03.25_05:13-attack_with_bert-tagspec-rnd-select:3.log',
    '03.25_05:11-attack_with_bert-tagspec-rnd-select:4.log',
    '03.25_05:14-attack_with_bert-tagspec-rnd-select:5.log',
    '03.25_05:14-attack_with_bert-tagspec-rnd-select:6.log',
    '03.25_05:14-attack_with_bert-tagspec-rnd-select:7.log'
]


def check_result():
    keys = None
    for log in log_files:
        with open(os.path.join(log_dir, log), "r") as f:
            logs = f.read().split('\n')[-10:]
            for line in logs:
                if line.startswith('{'):
                    result_dict = eval(line)
                    if keys is None:
                        keys = list(result_dict.keys())
                        print(keys)
                    print(log, end=' ')
                    for key in keys:
                        print(result_dict[key], end=' ')
                    print()

check_result()
