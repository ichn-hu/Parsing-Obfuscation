import os

work_dir = '$ROOT/v1zhu2/work'

tags = [
    ["NNP", "NNPS"],
    ["NN", "NNS"],
    ["JJ", "JJR", "JJS"],
    ["VB", "VBN", "VBD", "VBZ", "VBP", "VBG"],
    ["RB", "RBR", "RBS"],
    ["FW"],
    ["UH"]
]

result_dirs = [
    'build_allennlp_attacker-tagspec-use_pos-select:1-03.24_15:59',
    'build_allennlp_attacker-tagspec-use_pos-select:2-03.24_15:59',
    'build_allennlp_attacker-tagspec-use_pos-select:3-03.24_15:59',
    'build_allennlp_attacker-tagspec-use_pos-select:4-03.24_15:59',
    'build_allennlp_attacker-tagspec-use_pos-select:5-03.24_15:59',
    'build_allennlp_attacker-tagspec-use_pos-select:6-03.24_15:59',
    'build_allennlp_attacker-tagspec-use_pos-select:7-03.24_15:59',
    'build_allennlp_attacker-tagspec-rnd-use_pos-select:1-03.24_15:59',
    'build_allennlp_attacker-tagspec-rnd-use_pos-select:2-03.24_15:59',
    'build_allennlp_attacker-tagspec-rnd-use_pos-select:3-03.24_15:59',
    'build_allennlp_attacker-tagspec-rnd-use_pos-select:4-03.24_15:59',
    'build_allennlp_attacker-tagspec-rnd-use_pos-select:5-03.24_15:59',
    'build_allennlp_attacker-tagspec-rnd-use_pos-select:6-03.24_15:59',
    'build_allennlp_attacker-tagspec-rnd-use_pos-select:7-03.24_19:59',
]

def check_integrity():
    keys = ['tot']
    for tag in tags:
        keys += tag
    print(keys)
    for result_dir in result_dirs:
        mrr_result_path = os.path.join(work_dir, result_dir, 'mrr_result.txt')
        print(result_dir, end=' ')
        with open(mrr_result_path, "r") as f:
            result = eval(f.read())
        
        for key in keys:
            print(result[key] if key in result else 0, end=' ')
        print()


check_integrity()
