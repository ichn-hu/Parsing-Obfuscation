from typing import Dict, Optional, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.training.metrics import AttachmentScores
from allennlp.training.metrics import Metric
from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {'``', "''", ':', ',', '.', 'PU', 'PUNCT', 'SYM'}

@Model.register('mrr_metric')
class MeanRankReciprocalMetric(Metric):
    def __init__(self):
        self._reciprocal_rank = defaultdict(list)
        self._metadata = []
    
    def __call__(self,
                 predicted_distribution: torch.Tensor,
                 target_words: torch.Tensor,
                 target_postags: torch.Tensor,
                 vocab: Vocabulary):
        if predicted_distribution.numel() == 0:
            return
        numel, vocab_size = predicted_distribution.shape
        scores = predicted_distribution[torch.arange(numel), target_words]
        ranks = (predicted_distribution >= scores.reshape(numel, 1)).long().sum(dim=-1)
        # reciprocal_ranks = 1 / ranks
        for i in range(numel):
            # import ipdb
            # ipdb.set_trace()
            target_word = vocab.get_token_from_index(target_words[i].item())
            target_postag = vocab.get_token_from_index(target_postags[i].item(), 'pos')
            predicted_rank = ranks[i].item()
            self._metadata.append({
                "target_word": target_word,
                "target_postag": target_postag,
                "prediction_rank": predicted_rank
            })
            self._reciprocal_rank[target_postag].append(1 / predicted_rank)
    
    def get_metric(self, reset: bool = False):
        mrr = {}
        tot = 1e-10 # avoid dividing by zero
        sum_reciprocal_rank = 0.
        for pos, rr in self._reciprocal_rank.items():
            if pos.startswith('@'):
                continue
            mrr[pos] = sum(rr) / len(rr)
            tot += len(rr)
            sum_reciprocal_rank += sum(rr)
        mrr["tot"] = sum_reciprocal_rank / tot
        if reset:
            self._metadata = []
            self._reciprocal_rank = defaultdict(list)
        return mrr


@Model.register('attacker')
class PredictBackAttacker(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 pos_tag_embedding: Embedding = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        encoder_dim = encoder.get_output_dim()
        num_words = vocab.get_vocab_size('tokens')
        self.decoder = nn.Linear(encoder_dim, num_words)

        self._pos_tag_embedding = pos_tag_embedding or None
        dropout = 0.3
        input_dropout = 0.3
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self.loss = CrossEntropyLoss(reduction="elementwise_mean")
        self.mrr_metric = MeanRankReciprocalMetric()

    @overrides
    def forward(self,
            obf_words: Dict[str, torch.LongTensor],
            ori_words: Dict[str, torch.LongTensor],
            pos_tags: torch.LongTensor,
            obf_masks: torch.LongTensor,
            metadata) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.text_field_embedder(obf_words)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(obf_words)
        encoded_text = self.encoder(embedded_text_input, mask)
        encoded_text = self._dropout(encoded_text)
        decoded_prediction = self.decoder(encoded_text)
        predicted_distribution = decoded_prediction[obf_masks.byte()]
        if ori_words is not None and predicted_distribution.numel() != 0:
            # import ipdb
            # ipdb.set_trace()
            target_words = ori_words["tokens"][obf_masks.byte()]
            target_postags = pos_tags[obf_masks.byte()]
            self.mrr_metric(predicted_distribution, target_words, target_postags, self.vocab)
            loss = self.loss(predicted_distribution, target_words)
        else:
            target_words = None
            loss = torch.tensor(0., requires_grad=True) + 0 # then loss is not leaf variable

        output_dict = {
            "loss": loss,
            "predicted_distribution": predicted_distribution,
            "target_words": target_words
        }
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.mrr_metric.get_metric(reset)
