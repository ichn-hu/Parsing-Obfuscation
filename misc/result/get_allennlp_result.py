import os

work_dir = '/disk/ostrom/v1zhu2/work/lcpm'

result_dirs = [
    'load_constituency_parsing_model-ctx-sfnt:1-04.07_09:56',
    'load_constituency_parsing_model-ctx-sfnt:2-04.07_09:56',
    'load_constituency_parsing_model-ctx-sfnt:3-04.07_09:56',
    'load_constituency_parsing_model-ctx-sfnt:4-04.07_09:56',
    'load_constituency_parsing_model-ctx-sfnt:5-04.07_09:56',
    'load_constituency_parsing_model-ctx-sfnt:6-04.07_09:56',
    'load_constituency_parsing_model-ctx-sfnt:7-04.07_09:56',
    'load_constituency_parsing_model-rnd-sfnt:1-04.07_09:56',
    'load_constituency_parsing_model-rnd-sfnt:2-04.07_09:56',
    'load_constituency_parsing_model-rnd-sfnt:3-04.07_09:56',
    'load_constituency_parsing_model-rnd-sfnt:4-04.07_09:56',
    'load_constituency_parsing_model-rnd-sfnt:5-04.07_09:56',
    'load_constituency_parsing_model-rnd-sfnt:6-04.07_09:56',
    'load_constituency_parsing_model-rnd-sfnt:7-04.07_09:56',
]


def check_result():
    for res_dir in result_dirs:
        with open(os.path.join(work_dir, res_dir, 'result.json'), 'r') as f:
            result = eval(f.read())
            f1 = result['evalb_f1_measure']
            print(res_dir, f1)

check_result()