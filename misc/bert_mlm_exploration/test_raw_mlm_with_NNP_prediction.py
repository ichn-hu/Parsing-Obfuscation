import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

is_NNP = lambda pos: pos == 'NNP' or pos == 'NNPS'
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
model = BertForMaskedLM.from_pretrained('bert-base-cased')
model.eval()

def prediction_to_tokens(predictions):
    predicted_indexes = torch.argmax(predictions, dim=-1)
    tokens = []
    for predicted_sent_indexes in predicted_indexes:
        tokens.append(tokenizer.convert_ids_to_tokens([idx.item() for idx in predicted_sent_indexes]))
    return tokens


class SentenceInstance:
    def __init__(self, words, pos_tags):
        self.words = words
        self.pos_tags = pos_tags
        self.tokenized_words = tokenizer.tokenize(' '.join(words))
        self.len = len(words)

    def tokenized_match(self):
        la, lb = len(self.tokenized_words), len(self.words)
        # print(self.tokenized_words)
        # print(self.words)
        return la == lb

    def to_masked_sentence(self):
        sent = []
        token_ids = []
        token_spans = []
        mask_cnt = 0
        has_singleleton = False
        mask_converted = False
        for i in range(len(self.words)):
            token_spans.append(len(sent))
            tokenized_words = tokenizer.tokenize(self.words[i])
            if len(tokenized_words) == 1:
                has_singleleton = True

            if is_NNP(self.pos_tags[i]) and len(tokenized_words) == 1 and not mask_converted:
                # print(tokenized_words, self.words[i])
                mask_cnt += 1
                mask_converted = True
                for word in tokenized_words:
                    sent.append('[MASK]')
                    token_ids.append(i)
            else:
                for word in tokenized_words:
                    sent.append(word)
                    token_ids.append(i)
        token_spans.append(len(sent))
        word_ids = tokenizer.convert_tokens_to_ids(sent)

        return mask_cnt, sent, word_ids, self.words, token_ids, token_spans, has_singleleton

    def get_masked_num(self):
        cnt = 0
        for i in self.pos_tags:
            if is_NNP(i):
                cnt += 1
        return cnt

    def test_MLM(self):
        tokenized_text = self.to_masked_sentence()
        segments_ids = [0 for _ in tokenized_text]
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])   
        predictions = model(tokens_tensor, segments_tensors)
        print(self.tokenized_words)
        predicted_tokens = prediction_to_tokens(predictions)
        print(predicted_tokens)
   

def read_conll_raw(filepath):
    f = open(filepath, "r").read()
    sents = f.split("\n\n")
    sent_inst = []
    all_words = defaultdict(int)
    for sent in sents:
        tokens = sent.split("\n")
        words = []
        pos_tags = []
        for token in tokens:
            fields = token.split('\t')
            if len(fields) != 10:
                print(token)
                continue
            all_words[fields[1]] += 1
            words.append(fields[1])
            pos_tags.append(fields[4])
        sent_inst.append(SentenceInstance(words, pos_tags))
    # import ipdb
    # ipdb.set_trace()
    print(len((all_words.keys())))
    return sent_inst


def test_MLM():
    filepath = "$ROOT/Project/data/ptb/test.conll"
    sent_insts = read_conll_raw(filepath)
    tot = 0
    tot_masked = 0
    batch_size = 32
    mask_cnt = []
    cvt_words = []
    cvt_word_ids = []
    words = []
    token_ids = []
    token_spans = []
    sent_len = []
    predicted_words = []
    rcv_cnt = 0
    rcv_masked_cnt = 0
    rcv_pairs = []

    from tqdm import tqdm
    for sent_inst in tqdm(sent_insts):
        _mask_cnt, _cvt_words, _cvt_word_ids, _words, _token_ids, _token_spans, has_singleleton = sent_inst.to_masked_sentence()
        if _mask_cnt == 0 or not has_singleleton:
            continue

        tot_masked += _mask_cnt
        mask_cnt.append(_mask_cnt)
        cvt_words.append(_cvt_words)
        cvt_word_ids.append(_cvt_word_ids)
        words.append(_words)
        token_ids.append(_token_ids)
        token_spans.append(_token_spans)
        sent_len.append(len(_cvt_word_ids))

        prediction = model(torch.tensor([_cvt_word_ids]))
        _predicted_words = prediction_to_tokens(prediction)[0]
        predicted_words.append(_predicted_words)

        for i, word in enumerate(_words):
            left, right = _token_spans[i], _token_spans[i + 1]
            predicted_tokens = _predicted_words[left:right]
            predicted_word = ''.join([a.strip('#') for a in predicted_tokens])
            original_word = ''.join([a.strip('#') for a in tokenizer.tokenize(_words[i])])
            if _cvt_words[left] == '[MASK]':
                rcv_pairs.append((predicted_word, original_word))

            if predicted_word.lower() == original_word.lower():
                rcv_cnt += 1
                if _cvt_words[left] == '[MASK]':
                    rcv_masked_cnt += 1
                    print("Recovered {}".format(_words[i]))

    tot = sum(sent_len)
    import ipdb
    ipdb.set_trace()
    print("Recovered {}/{}/{:.2f}% {}/{}/{:.2f}%".format(rcv_cnt, tot, rcv_cnt / tot * 100,
        rcv_masked_cnt, tot_masked, rcv_masked_cnt / tot_masked * 100))



def test_MLM_simple():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenized input
    text = "Who was Jeff Dean ? Jeff Dean was a puppeteer"
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 6
    tokenized_text[masked_index] = '[MASK]'
    # assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']
    print(tokenized_text)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    # segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    # Predict all tokens
    predictions = model(tokens_tensor, segments_tensors)

    # confirm we were able to predict 'henson'
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)
    assert predicted_token == 'dean'
    

if __name__ == '__main__':
    test_MLM()

