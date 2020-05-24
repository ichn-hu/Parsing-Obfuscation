import os
from tqdm import tqdm
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
import config
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Instance:
    def __init__(self, raw_tokens, pos_tags, tokenizer):
        self.raw_tokens = raw_tokens
        self.pos_tags = pos_tags
        self.tokens = [tokenizer.tokenize(token) for token in raw_tokens]
        self.len = len(pos_tags)

    def get_masked_input(self, masked_index):
        input_tokens = []
        left = 0
        right = 0
        for i in range(self.len):
            if i == masked_index:
                left = len(input_tokens)
                input_tokens += ['[MASK]'] * len(self.tokens[i])
                targeted_tokens = self.tokens[i]
                right = len(input_tokens)
            else:
                input_tokens += self.tokens[i]
        return {"input_tokens": input_tokens,
                "masked_pos_tag": self.pos_tags[masked_index],
                "masked_token": self.tokens[masked_index],
                "masked_raw_token": self.raw_tokens[masked_index],
                "masked_index": masked_index,
                "targeted_tokens": targeted_tokens,
                "masked_index_range": (left, right),
                "segments_ids": [0] * len(input_tokens)}

def get_reciprocal_rank(distribution, target_index):
    score = distribution[target_index]
    rank = (distribution > score).long().sum().item() + 1
    return 1. / rank

def analysis(predictions, masked_inputs, tokenizer):
    batch_size, sentence_length, size_of_vocab = predictions.shape
    # (1245 * 32) * 30000 MRR 确实挺慢的啊 = =
    result = []
    for i in range(batch_size):
        reciprocal_rank = 0
        left, right = masked_inputs[i]["masked_index_range"]
        targeted_tokens = masked_inputs[i]["targeted_tokens"]
        targeted_indexes = tokenizer.convert_tokens_to_ids(targeted_tokens)
        for j in range(left, right):
            reciprocal_rank = max(reciprocal_rank, get_reciprocal_rank(predictions[i][j], targeted_indexes[j - left]))
        result.append((masked_inputs[i]["masked_pos_tag"], masked_inputs[i]["masked_token"], masked_inputs[i]["masked_raw_token"], reciprocal_rank))
    return result


def predict_on_instances(masked_inputs, tokenizer, model):
    max_len = max([len(inp["input_tokens"]) for inp in masked_inputs])
    indexed_tokens = [tokenizer.convert_tokens_to_ids(inp["input_tokens"] + ["[PAD]"] * (max_len - len(inp["input_tokens"]))) for inp in masked_inputs]
    segments_ids = [inp["segments_ids"] + [0] * (max_len - len(inp["segments_ids"])) for inp in masked_inputs]

    tokens_tensor = torch.tensor(indexed_tokens).to(config.cfg.device)
    segments_tensor = torch.tensor(segments_ids).to(config.cfg.device)

    predictions = model(tokens_tensor, segments_tensor)
    
    result = (predictions.to(torch.device('cpu')), masked_inputs)
    
    return result

def get_instances_from_conll(path, tokenizer):
    dataset_reader = UniversalDependenciesDatasetReader()
    dependency_instances = list(dataset_reader._read(path))
    logger.info("%d instances readed from %s", len(dependency_instances), path)
    insts = []
    for inst in tqdm(dependency_instances):
        insts.append(Instance(inst.fields['metadata']['words'], inst.fields['metadata']['pos'], tokenizer))

    return sorted(insts, key=lambda x: x.len, reverse=True)

def detailed_analysis(analysised_result):
    result = []
    for r in analysised_result:
        result += r
    
    num = len(result)
    tagdict = {}
    tot = 0
    for r in result:
        tot += r[-1]
        if r[0] in tagdict.keys():
            tagdict[r[0]].append(r[-1])
        else:
            tagdict[r[0]] = [r[-1]]
    
    tag_avg = {k: sum(tagdict[k]) / len(tagdict[k]) for k in tagdict.keys()}
    tot_avg = tot / num

    print(tot_avg)

    save_dir = config.cfg.ext.save_dir
    tag_avg['tot'] = tot_avg
    with open(os.path.join(save_dir, "bert_result.json"), "w") as f:
        f.write(str(tag_avg))
    print(tag_avg)

def attack_with_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data_path = os.environ.get("attack_file", config.cfg.ext.dev_path)
    insts = get_instances_from_conll(data_path, tokenizer)
    
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model = model.to(config.cfg.device)
    model.eval()
    masked_inputs = []

    for inst in insts:
        for i in range(inst.len):
            masked_inputs.append(inst.get_masked_input(i))

    batch_size = 32
    result = []
    analysised_result = []
    # masked_inputs = masked_inputs[:10 * batch_size]
    print("Evaluation started ", len(masked_inputs) / batch_size, " to go")
    with torch.no_grad():
        for i in tqdm(range(0, len(masked_inputs), batch_size)):
            batched_masked_inputs = masked_inputs[i: i + batch_size]
            result.append(predict_on_instances(batched_masked_inputs, tokenizer, model))
            if len(result) > 50:
                # print("Analysis started")
                for i in tqdm(result):
                    analysised_result.append(analysis(i[0], i[1], tokenizer))
                result = []

    if len(result) > 0:
        # print("Analysis started")
        for i in tqdm(result):
            analysised_result.append(analysis(i[0], i[1], tokenizer))
        result = []

    output_filename = os.path.join(config.cfg.ext.save_dir, "attacker_with_bert-analysised_result.ptr")
    with open(output_filename, "wb") as f:
        torch.save(analysised_result, f)
        logger.info("Analysised result saved to %s", output_filename)

    print(len(analysised_result))
    detailed_analysis(analysised_result)

def analysis_bert_attack_result():
    resume_result = os.environ.get("resume_result", "$ROOT/v1zhu2/work/attack_with_bert-02.27_13:46/attacker_with_bert-analysised_result.ptr")

    result = torch.load(resume_result)
    detailed_analysis(result)
