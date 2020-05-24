import torch
from .conll_allen import CoNLLUReader
from allennlp.data.dataset import Batch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator

def get_real_length(batch):
    """get_real_length
    Get real length list of the batch from Allen NLP's iterator
    """
    ret = [len(item.fields['tokens'].tokens) for item in batch.instances]
    return ret

def get_mask(batch):
    """get_mask
    Get token mask of the batch from Allen NLP's iterator
    """
    padding_length = batch.get_padding_lengths()['tokens']['num_tokens']
    real_lengths = get_real_length(batch)
    batch_size = len(batch.instances)
    ret = torch.ones(batch_size, padding_length)
    for i, rl in enumerate(real_lengths):
        ret[i, rl:] = 0
    return ret    


def BucketIteratorWrapper(instances, vocab, batch_size:int, shuffle=True):
    """
    Iterator wrapper for BucketIterator in allennlp.
    For training.
    """
    iterator = BucketIterator(sorting_keys=[("tags", "num_tokens"),], batch_size=batch_size)
    iterator.vocab = vocab
    for batch in iterator._create_batches(instances, shuffle=shuffle):
        padding_lengths = batch.get_padding_lengths()
        masks = get_mask(batch)
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict['tokens']['tokens']
        chars = tensor_dict['tokens']['chars']
        utags = tensor_dict['utags']
        tags = tensor_dict['tags']
        heads = tensor_dict['heads']
        rels = tensor_dict['rels']
        morphs = tensor_dict['morph']['morphs']
        subwords = tensor_dict['subword']['subwords']
        lengths = torch.LongTensor(get_real_length(batch))
        yield tokens, chars, utags, tags, lengths, masks, heads, rels, morphs, subwords


def MultipleInputIteratorCombiner(instances_list, vocab_list, iterator_generator, batch_size_per_dataset:int, shuffle=True):
    """
    Combine multiple data sources into one batch.
    This iterator will stop when the first dataset used up. Then it will pad tags/morph/subword/mask.
    Arguments:
        instances_list: a list of all input instances.
        vocab: vocab for generating index
        iterator_generator: the function to make a iterator
        batch_size_per_dataset: how many data items should be sampled from each instances object
        shuffle: shuffle the datasets or not

    Example:
        train_loader = MultipleInputIteratorCombiner([trainset1, trainset2,...], [vocab1, vocab2, ...],
         BucketIteratorWrapper, 32)
    """
    it_gen = lambda x, y: iterator_generator(x, y, batch_size_per_dataset, shuffle=shuffle)
    # chem the first instance is maximum instances
    for i, inst in enumerate(instances_list):
        if len(inst) > len(instances_list[0]):
            print("The first data is not the largest in MultipleInputIteratorCombiner")

    iterators_list = [it_gen(x, y) for x, y in zip(instances_list, vocab_list)]
    for batch0 in iterators_list[0]:
        batches = [batch0, ]
        # `batch0[2]` is the utags filed of `batch0`
        # by checking its length to get the size of this mini-batch
        # since $utags \in R^{B \times |T|}$, where B is the mini-batch size
        # The final batch may be smaller than `batch_size_per_dataset`
        # So just abadon them.
        if len(batch0[2]) < batch_size_per_dataset:
            continue
        for i in range(1, len(iterators_list)):
            it = iterators_list[i]
            while True:
                try:
                    b = next(it)
                    if len(b[2]) == batch_size_per_dataset:
                        break
                except StopIteration:
                    # when this iterator used up, create a new iterator
                    iterators_list[i] = it_gen(instances_list[i], vocab_list[i])
                    it = iterators_list[i]
                    b = next(it)
            batches.append(b)

        tokens = []
        chars =[]
        utags = []
        tags = []
        lengths = []
        masks = []
        heads = []
        rels = []
        morphs = []
        subwords = []

        # get lengths by checking the length of utags
        batches_len = [len(b[2][0]) for b in batches]
        max_padded_length = max(batches_len)
        for b in batches:
            token, char, utag, tag, length, mask, head, rel, morph, subword = b
            padded_length = len(utag[0])

            # pad zeros for utag/tag/morph/subword
            if padded_length < max_padded_length:
                padded_zeros = torch.zeros(batch_size_per_dataset, max_padded_length - padded_length)
                utag = torch.cat([utag, padded_zeros.type_as(utag)], dim=1)
                tag = torch.cat([tag, padded_zeros.type_as(tag)], dim=1)
                mask = torch.cat([mask, padded_zeros.type_as(mask)], dim=1)
                padded_zeros = torch.zeros(batch_size_per_dataset, max_padded_length - padded_length, morph.shape[2])
                morph = torch.cat([morph, padded_zeros.type_as(morph)], dim=1)
                padded_zeros = torch.zeros(batch_size_per_dataset, max_padded_length - padded_length, subword.shape[2])
                subword = torch.cat([subword, padded_zeros.type_as(subword)], dim=1)
            
            # pack data for yielding
            tokens.append(token)
            chars.append(char)
            utags.append(utag)
            tags.append(tag)
            lengths.append(length)
            masks.append(mask)
            heads.append(head)
            rels.append(rel)
            morphs.append(morph)
            subwords.append(subword)

        # reshape subwords and morphs at dim=2
        max_padded_dim2 = max([b.shape[2] for b in subwords])
        for i in range(len(subwords)):
            b = subwords[i]
            if b.shape[2] < max_padded_dim2:
                padded_zeros = torch.zeros(b.shape[0], b.shape[1], max_padded_dim2 - b.shape[2])
                b = torch.cat([b, padded_zeros.type_as(b)], dim=2)
                subwords[i] = b
        max_padded_dim2 = max([b.shape[2] for b in morphs])
        for i in range(len(morphs)):
            b = morphs[i]
            if b.shape[2] < max_padded_dim2:
                padded_zeros = torch.zeros(b.shape[0], b.shape[1], max_padded_dim2 - b.shape[2])
                b = torch.cat([b, padded_zeros.type_as(b)], dim=2)
                morphs[i] = b
        yield tokens, chars, utags, tags, lengths, masks, heads, rels, morphs, subwords


def BasicIteratorWrapper(instances, vocab, batch_size:int, shuffle=True):
    """
    Iterator wrapper for BasicIterator in allennlp.
    For validation.
    """
    iterator = BasicIterator(batch_size=batch_size)
    iterator.vocab = vocab
    for batch in iterator._create_batches(instances, shuffle=shuffle):
        for item in batch.instances:
            item.fields['tokens'].index(vocab)
            item.fields['lem_tokens'].index(vocab)
            item.fields['utags'].index(vocab)
            item.fields['tags'].index(vocab)
            item.fields['rels'].index(vocab)
            item.fields['morph'].index(vocab)
            item.fields['subword'].index(vocab)
        padding_lengths = batch.get_padding_lengths()
        
        masks = get_mask(batch)
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict['tokens']['tokens']
        chars = tensor_dict['tokens']['chars']
        utags = tensor_dict['utags']
        tags = tensor_dict['tags']
        heads = tensor_dict['heads']
        rels = tensor_dict['rels']
        morphs = tensor_dict['morph']['morphs']
        subwords = tensor_dict['subword']['subwords']
        lengths = torch.LongTensor(get_real_length(batch))
        yield tokens, chars, utags, tags, lengths, masks, heads, rels, morphs, subwords


def BasicTestIteratorWrapper(instances, vocab, batch_size: int, shuffle=True):
    """
    Iterator wrapper for BasicIterator in allennlp.
    For inference, which does not require `head` and `rels` fields.
    """
    iterator = BasicIterator(batch_size=batch_size)
    iterator.vocab = vocab
    for batch in iterator._create_batches(instances, shuffle=shuffle):
        original_tokens = []
        original_lem_tokens = []
        original_morphs = []
        for item in batch.instances:
            item.fields['tokens'].index(vocab)
            item.fields['lem_tokens'].index(vocab)
            item.fields['utags'].index(vocab)
            item.fields['tags'].index(vocab)
            item.fields['morph'].index(vocab)
            item.fields['subword'].index(vocab)
            original_tokens.append(item.fields['tokens'].tokens)
            original_lem_tokens.append(item.fields['lem_tokens'].tokens)
            original_morphs.append(item.fields['morph'].tokens)
        padding_lengths = batch.get_padding_lengths()
        masks = get_mask(batch)
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict['tokens']['tokens']
        chars = tensor_dict['tokens']['chars']
        utags = tensor_dict['utags']
        tags = tensor_dict['tags']
        morphs = tensor_dict['morph']['morphs']
        subwords = tensor_dict['subword']['subwords']
        lengths = torch.LongTensor(get_real_length(batch))
        yield tokens, chars, utags, tags, lengths, masks, morphs, subwords, original_tokens, original_lem_tokens, original_morphs

