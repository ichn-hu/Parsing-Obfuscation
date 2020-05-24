from typing import Dict, List, Tuple
import logging
import os
import tempfile
import subprocess
import shutil

from overrides import overrides
from nltk import Tree
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader # pylint: disable=no-name-in-module

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SpanField, SequenceLabelField, ListField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_EVALB_DIR = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "tools", "EVALB"))

@Metric.register("evalb_gai")
class EvalbBracketingScorer(Metric):
    """
    This class uses the external EVALB software for computing a broad range of metrics
    on parse trees. Here, we use it to compute the Precision, Recall and F1 metrics.
    You can download the source for EVALB from here: <http://nlp.cs.nyu.edu/evalb/>.

    Note that this software is 20 years old. In order to compile it on modern hardware,
    you may need to remove an ``include <malloc.h>`` statement in ``evalb.c`` before it
    will compile.

    AllenNLP contains the EVALB software, but you will need to compile it yourself
    before using it because the binary it generates is system dependent. To build it,
    run ``make`` inside the ``allennlp/tools/EVALB`` directory.

    Note that this metric reads and writes from disk quite a bit. You probably don't
    want to include it in your training loop; instead, you should calculate this on
    a validation set only.

    Parameters
    ----------
    evalb_directory_path : ``str``, required.
        The directory containing the EVALB executable.
    evalb_param_filename: ``str``, optional (default = "COLLINS.prm")
        The relative name of the EVALB configuration file used when scoring the trees.
        By default, this uses the COLLINS.prm configuration file which comes with EVALB.
        This configuration ignores POS tags and some punctuation labels.
    """
    def __init__(self,
                 evalb_directory_path: str = DEFAULT_EVALB_DIR,
                 evalb_param_filename: str = "COLLINS.prm") -> None:
        self._evalb_directory_path = evalb_directory_path
        self._evalb_program_path = os.path.join(evalb_directory_path, "evalb")
        self._evalb_param_path = os.path.join(evalb_directory_path, evalb_param_filename)


        self._header_line = ['ID', 'Len.', 'Stat.', 'Recal', 'Prec.', 'Bracket',
                             'gold', 'test', 'Bracket', 'Words', 'Tags', 'Accracy']

        self._correct_predicted_brackets = 0.0
        self._gold_brackets = 0.0
        self._predicted_brackets = 0.0

    @overrides
    def __call__(self, predicted_trees: List[Tree], gold_trees: List[Tree]) -> None: # type: ignore
        """
        Parameters
        ----------
        predicted_trees : ``List[Tree]``
            A list of predicted NLTK Trees to compute score for.
        gold_trees : ``List[Tree]``
            A list of gold NLTK Trees to use as a reference.
        """
        if not os.path.exists(self._evalb_program_path):
            logger.warning(f"EVALB not found at {self._evalb_program_path}.  Attempting to compile it.")
            EvalbBracketingScorer.compile_evalb(self._evalb_directory_path)

            # If EVALB executable still doesn't exist, raise an error.
            if not os.path.exists(self._evalb_program_path):
                compile_command = (f"python -c 'from allennlp.training.metrics import EvalbBracketingScorer; "
                                   f"EvalbBracketingScorer.compile_evalb(\"{self._evalb_directory_path}\")'")
                raise ConfigurationError(f"EVALB still not found at {self._evalb_program_path}. "
                                         "You must compile the EVALB scorer before using it."
                                         " Run 'make' in the '{}' directory or run: {}".format(
                                                 self._evalb_program_path, compile_command))
        tempdir = tempfile.mkdtemp()
        gold_path = os.path.join(tempdir, "gold.txt")
        predicted_path = os.path.join(tempdir, "predicted.txt")
        output_path = os.path.join(tempdir, "output.txt")
        with open(gold_path, "w") as gold_file:
            for tree in gold_trees:
                gold_file.write(f"{tree.pformat(margin=1000000)}\n")

        with open(predicted_path, "w") as predicted_file:
            for tree in predicted_trees:
                predicted_file.write(f"{tree.pformat(margin=1000000)}\n")

        command = f"{self._evalb_program_path} -p {self._evalb_param_path} " \
                  f"{gold_path} {predicted_path} > {output_path}"
        subprocess.run(command, shell=True, check=True)

        with open(output_path) as infile:
            for line in infile:
                stripped = line.strip().split()
                if len(stripped) == 12 and stripped != self._header_line:
                    # This line contains results for a single tree.
                    numeric_line = [float(x) for x in stripped]
                    self._correct_predicted_brackets += numeric_line[5]
                    self._gold_brackets += numeric_line[6]
                    self._predicted_brackets += numeric_line[7]

        shutil.rmtree(tempdir)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average precision, recall and f1.
        """
        recall = self._correct_predicted_brackets / self._gold_brackets if self._gold_brackets > 0 else 0.0
        precision = self._correct_predicted_brackets / self._predicted_brackets if self._gold_brackets > 0 else 0.0
        f1_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        if reset:
            self.reset()
        return {"evalb_recall": recall, "evalb_precision": precision, "evalb_f1_measure": f1_measure}

    @overrides
    def reset(self):
        self._correct_predicted_brackets = 0.0
        self._gold_brackets = 0.0
        self._predicted_brackets = 0.0

    @staticmethod
    def compile_evalb(evalb_directory_path: str = DEFAULT_EVALB_DIR):
        logger.info(f"Compiling EVALB by running make in {evalb_directory_path}.")
        os.system("cd {} && make && cd ../../../".format(evalb_directory_path))

    @staticmethod
    def clean_evalb(evalb_directory_path: str = DEFAULT_EVALB_DIR):
        os.system("rm {}".format(os.path.join(evalb_directory_path, "evalb")))


@DatasetReader.register("ptb_trees_gai")
class PennTreeBankConstituencySpanDatasetReader(DatasetReader):
    """
    Reads constituency parses from the WSJ part of the Penn Tree Bank from the LDC.
    This ``DatasetReader`` is designed for use with a span labelling model, so
    it enumerates all possible spans in the sentence and returns them, along with gold
    labels for the relevant spans present in a gold tree, if provided.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    use_pos_tags : ``bool``, optional, (default = ``True``)
        Whether or not the instance should contain gold POS tags
        as a field.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    label_namespace_prefix : ``str``, optional, (default = ``""``)
        Prefix used for the label namespace.  The ``span_labels`` will use
        namespace ``label_namespace_prefix + 'labels'``, and if using POS
        tags their namespace is ``label_namespace_prefix + pos_label_namespace``.
    pos_label_namespace : ``str``, optional, (default = ``"pos"``)
        The POS tag namespace is ``label_namespace_prefix + pos_label_namespace``.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_pos_tags: bool = True,
                 lazy: bool = False,
                 label_namespace_prefix: str = "",
                 pos_label_namespace: str = "pos") -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._use_pos_tags = use_pos_tags
        self._label_namespace_prefix = label_namespace_prefix
        self._pos_label_namespace = pos_label_namespace

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        directory, filename = os.path.split(file_path)
        logger.info("Reading instances from lines in file at: %s", file_path)
        for parse in BracketParseCorpusReader(root=directory, fileids=[filename]).parsed_sents():

            self._strip_functional_tags(parse)
            # This is un-needed and clutters the label space.
            # All the trees also contain a root S node.
            if parse.label() == "VROOT":
                parse = parse[0]
            pos_tags = [x[1] for x in parse.pos()] if self._use_pos_tags else None
            yield self.text_to_instance(parse.leaves(), pos_tags, parse)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         pos_tags: List[str] = None,
                         gold_tree: Tree = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        pos_tags ``List[str]``, optional, (default = None).
            The POS tags for the words in the sentence.
        gold_tree : ``Tree``, optional (default = None).
            The gold parse tree to create span labels from.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence.
            pos_tags : ``SequenceLabelField``
                The POS tags of the words in the sentence.
                Only returned if ``use_pos_tags`` is ``True``
            spans : ``ListField[SpanField]``
                A ListField containing all possible subspans of the
                sentence.
            span_labels : ``SequenceLabelField``, optional.
                The constituency tags for each of the possible spans, with
                respect to a gold parse tree. If a span is not contained
                within the tree, a span will have a ``NO-LABEL`` label.
            gold_tree : ``MetadataField(Tree)``
                The gold NLTK parse tree for use in evaluation.
        """
        # pylint: disable=arguments-differ
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        pos_namespace = self._label_namespace_prefix + self._pos_label_namespace
        if self._use_pos_tags and pos_tags is not None:
            pos_tag_field = SequenceLabelField(pos_tags, text_field,
                                               label_namespace=pos_namespace)
            fields["pos_tags"] = pos_tag_field
        elif self._use_pos_tags:
            raise ConfigurationError("use_pos_tags was set to True but no gold pos"
                                     " tags were passed to the dataset reader.")
        spans: List[Field] = []
        gold_labels = []

        if gold_tree is not None:
            gold_spans: Dict[Tuple[int, int], str] = {}
            self._get_gold_spans(gold_tree, 0, gold_spans)

        else:
            gold_spans = None
        for start, end in enumerate_spans(tokens):
            spans.append(SpanField(start, end, text_field))

            if gold_spans is not None:
                if (start, end) in gold_spans.keys():
                    gold_labels.append(gold_spans[(start, end)])
                else:
                    gold_labels.append("NO-LABEL")

        metadata = {"tokens": tokens}
        if gold_tree:
            metadata["gold_tree"] = gold_tree
        if self._use_pos_tags:
            metadata["pos_tags"] = pos_tags

        fields["metadata"] = MetadataField(metadata)

        span_list_field: ListField = ListField(spans)
        fields["spans"] = span_list_field
        if gold_tree is not None:
            fields["span_labels"] = SequenceLabelField(gold_labels,
                                                       span_list_field,
                                                       label_namespace=self._label_namespace_prefix + "labels")
        return Instance(fields)

    def _strip_functional_tags(self, tree: Tree) -> None:
        """
        Removes all functional tags from constituency labels in an NLTK tree.
        We also strip off anything after a =, - or | character, because these
        are functional tags which we don't want to use.

        This modification is done in-place.
        """
        clean_label = tree.label().split("=")[0].split("-")[0].split("|")[0]
        tree.set_label(clean_label)
        for child in tree:
            if not isinstance(child[0], str):
                self._strip_functional_tags(child)

    def _get_gold_spans(self, # pylint: disable=arguments-differ
                        tree: Tree,
                        index: int,
                        typed_spans: Dict[Tuple[int, int], str]) -> int:
        """
        Recursively construct the gold spans from an nltk ``Tree``.
        Labels are the constituents, and in the case of nested constituents
        with the same spans, labels are concatenated in parent-child order.
        For example, ``(S (NP (D the) (N man)))`` would have an ``S-NP`` label
        for the outer span, as it has both ``S`` and ``NP`` label.
        Spans are inclusive.

        TODO(Mark): If we encounter a gold nested labelling at test time
        which we haven't encountered, we won't be able to run the model
        at all.

        Parameters
        ----------
        tree : ``Tree``, required.
            An NLTK parse tree to extract spans from.
        index : ``int``, required.
            The index of the current span in the sentence being considered.
        typed_spans : ``Dict[Tuple[int, int], str]``, required.
            A dictionary mapping spans to span labels.

        Returns
        -------
        typed_spans : ``Dict[Tuple[int, int], str]``.
            A dictionary mapping all subtree spans in the parse tree
            to their constituency labels. POS tags are ignored.
        """
        # NLTK leaves are strings.
        if isinstance(tree[0], str):
            # The "length" of a tree is defined by
            # NLTK as the number of children.
            # We don't actually want the spans for leaves, because
            # their labels are POS tags. Instead, we just add the length
            # of the word to the end index as we iterate through.
            end = index + len(tree)
        else:
            # otherwise, the tree has children.
            child_start = index
            for child in tree:
                # typed_spans is being updated inplace.
                end = self._get_gold_spans(child, child_start, typed_spans)
                child_start = end
            # Set the end index of the current span to
            # the last appended index - 1, as the span is inclusive.
            span = (index, end - 1)
            current_span_label = typed_spans.get(span)
            if current_span_label is None:
                # This span doesn't have nested labels, just
                # use the current node's label.
                typed_spans[span] = tree.label()
            else:
                # This span has already been added, so prepend
                # this label (as we are traversing the tree from
                # the bottom up).
                typed_spans[span] = tree.label() + "-" + current_span_label

        return end
