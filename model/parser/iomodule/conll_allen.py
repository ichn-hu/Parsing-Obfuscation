from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Tokenizer, Token

import pdb

import numpy as np
import torch

import spacy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def _is_divider(line: str) -> bool:
    line = line.strip()
    if line.startswith("#") or line == "":
        return True
    return False
    # return not line or line == """-DOCSTART- -X- -X- O"""

_VALID_LABELS = {'ner', 'pos', 'chunk'}


# @DatasetReader.register("conll2003")
class CoNLLUReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    WORD POS-TAG CHUNK-TAG NER-TAG
    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.
    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.
    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_labels``, ``chunk_labels``, ``ner_labels``.
        If you want to use one of the labels as a `feature` in your model, it should be
        specified here.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False,
                 pre_params: Dict = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in _VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in _VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.pre_params = pre_params

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        def keep_follow_idx_list(inp_list, idx_list):
            out_list = []
            for i in idx_list:
                out_list.append(inp_list[i])
            return out_list

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)


            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip(" ").split("\t") for line in lines]
                    new_fields = []
                    for field in fields:
                        try:
                            int(field[6])
                            # if torch.rand(1) > 0.5:
                            #     field[6] = np.random.randint(low=0, high=len(fields))
                            new_fields.append(field)
                        except:
                            pass
                    if len(fields)> 0 and len(new_fields) == 0:
                        raise RuntimeError("No head detected. Is {} contains ground truth?".format(file_path))
                    fields = new_fields

                    # unzipping trick returns tuples, but our Fields need lists
                    id, word, word2, utag, tag, morph, head, rel, _, _\
                        = [list(field) for field in zip(*fields)]
                     
                    # keep list: ignore the tokens with '_' head
                    keep_list = []
                    for i, id_notation in enumerate(id):
                        if (not '-' in id_notation) and (not '.' in id_notation):
                            keep_list.append(i)
                    id = keep_follow_idx_list(id, keep_list)
                    word = keep_follow_idx_list(word, keep_list)
                    word2 = keep_follow_idx_list(word2, keep_list)
                    tag = keep_follow_idx_list(tag, keep_list)
                    utag = keep_follow_idx_list(utag, keep_list)
                    head = keep_follow_idx_list(head, keep_list)
                    rel = keep_follow_idx_list(rel, keep_list)
                    morph = keep_follow_idx_list(morph, keep_list)
                    word = ['<root>'] + word
                    tag = ['<root>'] + tag
                    utag = ['<root>'] + utag
                    morph = ['<root>'] + morph

                    try:
                        head = [int(t) for t in head]
                    except:
                        print(head)
                        print(word)
                        pdb.set_trace()

                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in word]
                    sequence = TextField(tokens, token_indexers={"tokens": SingleIdTokenIndexer("tokens"),
                                                                 "chars": TokenCharactersIndexer("chars")})
                    instance_fields: Dict[str, Field] = {'tokens': sequence}
                    lem_tokens = [Token(token) for token in word2]

                    instance_fields['lem_tokens'] = TextField(lem_tokens, self._token_indexers)
                    instance_fields['subword'] = TextField(tokens, token_indexers={'subwords': TokenSubwordIndexer("subwords")})
                    instance_fields['utags'] = SequenceLabelField(utag, sequence, "utags")
                    instance_fields['tags'] = SequenceLabelField(tag, sequence, "tags")

                    morph = [Token(token) for token in morph]
                    instance_fields['morph'] = \
                        TextField(morph, token_indexers={'tokens': SingleIdTokenIndexer(),
                                                         "morphs": TokenMorphIndexer("morphs")})
                    instance_fields['heads'] = SequenceLabelFieldCustom(head, sequence, "head_labels")
                    instance_fields['rels'] = SequenceLabelFieldCustom(rel, sequence, "rel_labels")

                    yield Instance(instance_fields)

    def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        return Instance({'tokens': TextField(tokens, token_indexers=self._token_indexers)})

    @classmethod
    def from_params(cls, params: Params) -> 'CoNLLUReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        tag_label = params.pop('tag_label', None)
        feature_labels = params.pop('feature_labels', ())
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return CoNLLUReader(token_indexers=token_indexers,
                            tag_label=tag_label,
                            feature_labels=feature_labels,
                            lazy=lazy)


class CoNLLUTestReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    WORD POS-TAG CHUNK-TAG NER-TAG
    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.
    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.
    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_labels``, ``chunk_labels``, ``ner_labels``.
        If you want to use one of the labels as a `feature` in your model, it should be
        specified here.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in _VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in _VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        def keep_follow_idx_list(inp_list, idx_list):
            out_list = []
            for i in idx_list:
                out_list.append(inp_list[i])
            return out_list

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip(" ").split("\t") for line in lines]

                    # unzipping trick returns tuples, but our Fields need lists
                    id, word, word2, utag, tag, morph, head, rel, _, _\
                        = [list(field) for field in zip(*fields)]

                    # keep list: in testing, all tokens should be kept
                    keep_list = []
                    for i, id_notation in enumerate(id):
                        if (not '-' in id_notation) and (not '.' in id_notation):
                            keep_list.append(i)
                    id = keep_follow_idx_list(id, keep_list)
                    word = keep_follow_idx_list(word, keep_list)
                    word2 = keep_follow_idx_list(word2, keep_list)
                    tag = keep_follow_idx_list(tag, keep_list)
                    utag = keep_follow_idx_list(utag, keep_list)
                    morph = keep_follow_idx_list(morph, keep_list)
                    word = ['<root>'] + word
                    word2 = ['<root>'] + word2
                    tag = ['<root>'] + tag
                    utag = ['<root>'] + utag
                    morph = ['<root>'] + morph

                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in word]

                    sequence = TextField(tokens, token_indexers={"tokens": SingleIdTokenIndexer("tokens"),
                                                                 "chars": TokenCharactersIndexer("chars")})
                    instance_fields: Dict[str, Field] = {'tokens': sequence}
                    lem_tokens = [Token(token) for token in word2]

                    instance_fields['lem_tokens'] = TextField(lem_tokens, self._token_indexers)
                    instance_fields['subword'] = TextField(tokens, token_indexers={'subwords': TokenSubwordIndexer("subwords")})
                    instance_fields['utags'] = SequenceLabelField(utag, sequence, "utags")
                    instance_fields['tags'] = SequenceLabelField(tag, sequence, "tags")

                    morph = [Token(token) for token in morph]
                    instance_fields['morph'] = \
                        TextField(morph, token_indexers={'tokens': SingleIdTokenIndexer(),
                                                         "morphs": TokenMorphIndexer("morphs")})

                    yield Instance(instance_fields)

    def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        return Instance({'tokens': TextField(tokens, token_indexers=self._token_indexers)})

    @classmethod
    def from_params(cls, params: Params) -> 'CoNLLUReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        tag_label = params.pop('tag_label', None)
        feature_labels = params.pop('feature_labels', ())
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return CoNLLUReader(token_indexers=token_indexers,
                            tag_label=tag_label,
                            feature_labels=feature_labels,
                            lazy=lazy)


from typing import Dict, List, Union, Set
import logging
import textwrap

from overrides import overrides
import torch
from torch.autograd import Variable

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SequenceLabelFieldCustom(Field[torch.Tensor]):
    """
    A ``SequenceLabelField`` assigns a categorical label to each element in a
    :class:`~allennlp.data.fields.sequence_field.SequenceField`.
    Because it's a labeling of some other field, we take that field as input here, and we use it to
    determine our padding and other things.

    This field will get converted into a list of integer class ids, representing the correct class
    for each element in the sequence.

    Parameters
    ----------
    labels : ``Union[List[str], List[int]]``
        A sequence of categorical labels, encoded as strings or integers.  These could be POS tags like [NN,
        JJ, ...], BIO tags like [B-PERS, I-PERS, O, O, ...], or any other categorical tag sequence. If the
        labels are encoded as integers, they will not be indexed using a vocab.
    sequence_field : ``SequenceField``
        A field containing the sequence that this ``SequenceLabelField`` is labeling.  Most often, this is a
        ``TextField``, for tagging individual tokens in a sentence.
    label_namespace : ``str``, optional (default='labels')
        The namespace to use for converting tag strings into integers.  We convert tag strings to
        integers for you, and this parameter tells the ``Vocabulary`` object which mapping from
        strings to integers to use (so that "O" as a tag doesn't get the same id as "O" as a word).
    """
    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()

    def __init__(self,
                 labels: Union[List[str], List[int]],
                 sequence_field: SequenceField,
                 label_namespace: str = 'labels') -> None:
        self.labels = labels
        self.sequence_field = sequence_field
        self._label_namespace = label_namespace
        self._indexed_labels = None
        self._maybe_warn_for_namespace(label_namespace)
        # if len(labels) != sequence_field.sequence_length():
        #     raise ConfigurationError("Label length and sequence length "
        #                              "don't match: %d and %d" % (len(labels), sequence_field.sequence_length()))

        if all([isinstance(x, int) for x in labels]):
            self._indexed_labels = labels

        elif not all([isinstance(x, str) for x in labels]):
            raise ConfigurationError("SequenceLabelFields must be passed either all "
                                     "strings or all ints. Found labels {} with "
                                     "types: {}.".format(labels, [type(x) for x in labels]))

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (self._label_namespace.endswith("labels") or self._label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning("Your label namespace was '%s'. We recommend you use a namespace "
                               "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                               "default to your vocabulary.  See documentation for "
                               "`non_padded_namespaces` parameter in Vocabulary.",
                               self._label_namespace)
                self._already_warned_namespaces.add(label_namespace)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._indexed_labels is None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1  # type: ignore

    @overrides
    def index(self, vocab: Vocabulary):
        if self._indexed_labels is None:
            self._indexed_labels = [vocab.get_token_index(label, self._label_namespace)  # type: ignore
                                    for label in self.labels]

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # return {'num_tokens': self.sequence_field.sequence_length()}
        return {'num_tokens': len(self.labels)}

    @overrides
    def as_tensor(self,
                  padding_lengths: Dict[str, int],
                  cuda_device: int = -1,
                  for_training: bool = True) -> torch.Tensor:
        desired_num_tokens = padding_lengths['num_tokens']
        padded_tags = pad_sequence_to_length(self._indexed_labels, desired_num_tokens)
        tensor = Variable(torch.LongTensor(padded_tags), volatile=not for_training)
        return tensor if cuda_device == -1 else tensor.cuda(cuda_device)

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        # pylint: disable=protected-access
        sequence_label_field = SequenceLabelField([], self.sequence_field.empty_field())
        sequence_label_field._indexed_labels = []
        return sequence_label_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        formatted_labels = "".join(["\t\t" + labels + "\n"
                                    for labels in textwrap.wrap(repr(self.labels), 100)])
        return f"SequenceLabelField of length {length} with " \
               f"labels:\n {formatted_labels} \t\tin namespace: '{self._label_namespace}'."


class SubwordTokenizer(Tokenizer):
    """
    A ``CharacterTokenizer`` splits strings into character tokens.

    Parameters
    ----------
    byte_encoding : str, optional (default=``None``)
        If not ``None``, we will use this encoding to encode the string as bytes, and use the byte
        sequence as characters, instead of the unicode characters in the python string.  E.g., the
        character 'á' would be a single token if this option is ``None``, but it would be two
        tokens if this option is set to ``"utf-8"``.

        If this is not ``None``, ``tokenize`` will return a ``List[int]`` instead of a
        ``List[str]``, and we will bypass the vocabulary in the ``TokenIndexer``.
    lowercase_characters : ``bool``, optional (default=``False``)
        If ``True``, we will lowercase all of the characters in the text before doing any other
        operation.  You probably do not want to do this, as character vocabularies are generally
        not very large to begin with, but it's an option if you really want it.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.  If
        using byte encoding, this should actually be a ``List[int]``, not a ``List[str]``.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.  If using byte
        encoding, this should actually be a ``List[int]``, not a ``List[str]``.
    """
    def __init__(self,
                 byte_encoding: str = None,
                 lowercase_characters: bool = False,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._byte_encoding = byte_encoding
        self._lowercase_characters = lowercase_characters
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]

    @overrides
    def tokenize(self, text: str) -> List[Token]: # MODIFY
        if self._lowercase_characters:
            text = text.lower()
        if text == '<root>':
            return [Token(text)]
        text = '<' + text + '>'
        tokens = [text[i:i + 3] for i in range(len(text) - 2)]

        tokens = [Token(token) for token in tokens]
        return tokens

    @classmethod
    def from_params(cls, params: Params) -> 'SubwordTokenizer':
        byte_encoding = params.pop('byte_encoding', None)
        lowercase_characters = params.pop('lowercase_characters', False)
        start_tokens = params.pop('start_tokens', None)
        end_tokens = params.pop('end_tokens', None)
        params.assert_empty(cls.__name__)
        return cls(byte_encoding=byte_encoding,
                   lowercase_characters=lowercase_characters,
                   start_tokens=start_tokens,
                   end_tokens=end_tokens)


class TokenSubwordIndexer(TokenIndexer[List[int]]):
    """
    This :class:`TokenIndexer` represents tokens as lists of character indices.

    Parameters
    ----------
    namespace : ``str``, optional (default=``token_characters``)
        We will use this namespace in the :class:`Vocabulary` to map the characters in each token
        to indices.
    character_tokenizer : ``CharacterTokenizer``, optional (default=``CharacterTokenizer()``)
        We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        ``CharacterTokenizer`` with its default parameters, which uses unicode characters and
        retains casing.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'token_subword',
                 character_tokenizer: SubwordTokenizer = SubwordTokenizer()) -> None:
        self._namespace = namespace
        self._character_tokenizer = character_tokenizer

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError('TokenCharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            # If `text_id` is set on the character token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this character.
            counter[self._namespace][character.text] += 1

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> List[int]:
        indices = []
        if token.text is None:
            raise ConfigurationError('TokenCharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            if getattr(character, 'text_id', None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just
                # use this id instead.
                index = character.text_id
            else:
                index = vocabulary.get_token_index(character.text, self._namespace)
            indices.append(index)
        return indices

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        return {'num_token_characters': len(token)}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    @overrides
    def pad_token_sequence(self,
                           tokens: List[List[int]],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[List[int]]:
        # Pad the tokens.
        padded_tokens = pad_sequence_to_length(tokens, desired_num_tokens, default_value=self.get_padding_token)

        # Pad the characters within the tokens.
        desired_token_length = padding_lengths['num_token_characters']
        longest_token: List[int] = max(tokens, key=len, default=[])
        padding_value = 0
        if desired_token_length > len(longest_token):
            # Since we want to pad to greater than the longest token, we add a
            # "dummy token" so we can take advantage of the fast implementation of itertools.zip_longest.
            padded_tokens.append([padding_value] * desired_token_length)
        # pad the list of lists to the longest sublist, appending 0's
        padded_tokens = list(zip(*itertools.zip_longest(*padded_tokens, fillvalue=padding_value)))
        if desired_token_length > len(longest_token):
            # Removes the "dummy token".
            padded_tokens.pop()
        # Truncates all the tokens to the desired length, and return the result.
        return [list(token[:desired_token_length]) for token in padded_tokens]

    @classmethod
    def from_params(cls, params: Params) -> 'TokenCharactersIndexer':
        """
        Parameters
        ----------
        namespace : ``str``, optional (default=``token_characters``)
            We will use this namespace in the :class:`Vocabulary` to map the characters in each token
            to indices.
        character_tokenizer : ``Params``, optional (default=``Params({})``)
            We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
            options for byte encoding and other things.  These parameters get passed to the character
            tokenizer.  The default is to use unicode characters and to retain casing.
        """
        namespace = params.pop('namespace', 'token_characters')
        character_tokenizer_params = params.pop('character_tokenizer', {})
        character_tokenizer = MorphTokenizer.from_params(character_tokenizer_params)
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace, character_tokenizer=character_tokenizer)


"""
morph tokenizer
"""


class MorphTokenizer(Tokenizer):
    """
    A ``CharacterTokenizer`` splits strings into character tokens.

    Parameters
    ----------
    byte_encoding : str, optional (default=``None``)
        If not ``None``, we will use this encoding to encode the string as bytes, and use the byte
        sequence as characters, instead of the unicode characters in the python string.  E.g., the
        character 'á' would be a single token if this option is ``None``, but it would be two
        tokens if this option is set to ``"utf-8"``.

        If this is not ``None``, ``tokenize`` will return a ``List[int]`` instead of a
        ``List[str]``, and we will bypass the vocabulary in the ``TokenIndexer``.
    lowercase_characters : ``bool``, optional (default=``False``)
        If ``True``, we will lowercase all of the characters in the text before doing any other
        operation.  You probably do not want to do this, as character vocabularies are generally
        not very large to begin with, but it's an option if you really want it.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.  If
        using byte encoding, this should actually be a ``List[int]``, not a ``List[str]``.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.  If using byte
        encoding, this should actually be a ``List[int]``, not a ``List[str]``.
    """
    def __init__(self,
                 byte_encoding: str = None,
                 lowercase_characters: bool = False,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._byte_encoding = byte_encoding
        self._lowercase_characters = lowercase_characters
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]

    @overrides
    def tokenize(self, text: str) -> List[Token]: # MODIFY
        if self._lowercase_characters:
            text = text.lower()
        if text == '<root>':
            return [Token(text)]

        tokens = text.split('|')
        tokens = [Token(token) for token in tokens]
        return tokens

    @classmethod
    def from_params(cls, params: Params) -> 'MorphTokenizer':
        byte_encoding = params.pop('byte_encoding', None)
        lowercase_characters = params.pop('lowercase_characters', False)
        start_tokens = params.pop('start_tokens', None)
        end_tokens = params.pop('end_tokens', None)
        params.assert_empty(cls.__name__)
        return cls(byte_encoding=byte_encoding,
                   lowercase_characters=lowercase_characters,
                   start_tokens=start_tokens,
                   end_tokens=end_tokens)


class TokenMorphIndexer(TokenIndexer[List[int]]):
    """
    This :class:`TokenIndexer` represents tokens as lists of character indices.

    Parameters
    ----------
    namespace : ``str``, optional (default=``token_characters``)
        We will use this namespace in the :class:`Vocabulary` to map the characters in each token
        to indices.
    character_tokenizer : ``CharacterTokenizer``, optional (default=``CharacterTokenizer()``)
        We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        ``CharacterTokenizer`` with its default parameters, which uses unicode characters and
        retains casing.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'token_morphs',
                 character_tokenizer: MorphTokenizer = MorphTokenizer()) -> None:
        self._namespace = namespace
        self._character_tokenizer = character_tokenizer

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError('TokenCharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            # If `text_id` is set on the character token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this character.
            counter[self._namespace][character.text] += 1

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> List[int]:
        indices = []
        if token.text is None:
            raise ConfigurationError('TokenCharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            if getattr(character, 'text_id', None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just
                # use this id instead.
                index = character.text_id
            else:
                index = vocabulary.get_token_index(character.text, self._namespace)
            indices.append(index)
        return indices

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        return {'num_token_characters': len(token)}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    @overrides
    def pad_token_sequence(self,
                           tokens: List[List[int]],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[List[int]]:
        # Pad the tokens.
        padded_tokens = pad_sequence_to_length(tokens, desired_num_tokens,
                                               default_value=self.get_padding_token)

        # Pad the characters within the tokens.
        desired_token_length = padding_lengths['num_token_characters']
        longest_token: List[int] = max(tokens, key=len, default=[])
        padding_value = 0
        if desired_token_length > len(longest_token):
            # Since we want to pad to greater than the longest token, we add a
            # "dummy token" so we can take advantage of the fast implementation of itertools.zip_longest.
            padded_tokens.append([padding_value] * desired_token_length)
        # pad the list of lists to the longest sublist, appending 0's
        padded_tokens = list(zip(*itertools.zip_longest(*padded_tokens, fillvalue=padding_value)))
        if desired_token_length > len(longest_token):
            # Removes the "dummy token".
            padded_tokens.pop()
        # Truncates all the tokens to the desired length, and return the result.
        return [list(token[:desired_token_length]) for token in padded_tokens]

    @classmethod
    def from_params(cls, params: Params) -> 'TokenCharactersIndexer':
        """
        Parameters
        ----------
        namespace : ``str``, optional (default=``token_characters``)
            We will use this namespace in the :class:`Vocabulary` to map the characters in each token
            to indices.
        character_tokenizer : ``Params``, optional (default=``Params({})``)
            We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
            options for byte encoding and other things.  These parameters get passed to the character
            tokenizer.  The default is to use unicode characters and to retain casing.
        """
        namespace = params.pop('namespace', 'token_characters')
        character_tokenizer_params = params.pop('character_tokenizer', {})
        character_tokenizer = MorphTokenizer.from_params(character_tokenizer_params)
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace, character_tokenizer=character_tokenizer)


class CoNLLWriter:
    def __init__(self, output_file, vocab):
        self.file_handler = open(output_file, 'w')
        self.vocab = vocab

    def close(self):
        self.file_handler.close()

    def _map_idx_to_tokens(self, idx_tensor, namespace):
        idx_list = idx_tensor.tolist()
        ret = []
        for l in idx_list:
            entry = []
            for x in l:
                entry.append(self.vocab.get_token_from_index(x, namespace=namespace))
            ret.append(entry)
        return ret
    
    def write(self, words, utags, tags, heads, labels,
              original_tokens, original_lem_tokens, original_morph):
        utags = self._map_idx_to_tokens(utags, 'utags')
        tags = self._map_idx_to_tokens(tags, 'tags')
        heads = heads.tolist()
        rels = self._map_idx_to_tokens(labels, 'rel_labels')
        batch_size = len(words)
        for i in range(batch_size):
            w = original_tokens[i][1:]
            lw = original_lem_tokens[i][1:]
            m = original_morph[i][1:]
            ut = utags[i][1:]
            t = tags[i][1:]
            h = heads[i]
            r = rels[i]
            for j in range(len(w)):
                self.file_handler.write("%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t_\t_\n" %
                                        (j+1, w[j], lw[j].text.lower(), ut[j], t[j], m[j], h[j], r[j]))
            self.file_handler.write("\n")

    def write_without_vocab(self, words, utags, tags, heads, labels, lem_tokens, morphs):
        """
        All inputs should be list[str] or torch tensor.
        """
        batch_size = len(words)
        for i in range(batch_size):
            w = words[i][1:]
            lw = lem_tokens[i]
            ut = utags[i][1:]
            t = tags[i][1:]
            h = heads[i]
            r = labels[i]
            m = morphs[i][1:]
            for j in range(len(w)):
                self.file_handler.write("%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t_\t_\n" %
                                        (j+1, w[j], lw[j].text.lower(), ut[j], t[j], m[j], h[j], r[j]))
            self.file_handler.write("\n")

