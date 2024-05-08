from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Union, Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field

from transformers import PretrainedConfig


class QuoteTypePredictorConfig(PretrainedConfig):
    def __init__(
        self,
        pretrained_model_name_or_path=None,
        cache_dir=None,
        revision="main",
        use_auth_token=False,
        hidden_dropout_prob=0,
        max_span_width=30,
        use_span_width_embedding=False,
        linear_size=32,
        init_temperature=0.07,
        start_loss_weight=0.2,
        end_loss_weight=0.2,
        span_loss_weight=0.6,
        threshold_loss_weight=0.5,
        ner_loss_weight=0.5,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.cache_dir = cache_dir
        self.revision = revision
        self.use_auth_token = use_auth_token
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_span_width = max_span_width
        self.use_span_width_embedding = use_span_width_embedding
        self.linear_size = linear_size
        self.init_temperature = init_temperature
        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.span_loss_weight = span_loss_weight
        self.threshold_loss_weight = threshold_loss_weight
        self.ner_loss_weight = ner_loss_weight


class QuoteTypePredictor(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            cache_dir=config.cache_dir,
            revision=config.revision,
            use_auth_token=config.use_auth_token,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.hf_config = hf_config
        self.config.pruned_heads = hf_config.pruned_heads
        self.dropout = torch.nn.Dropout(hf_config.hidden_dropout_prob)

        if config.use_span_width_embedding:
            self.span_linear = torch.nn.Linear(
                hf_config.hidden_size * 2 + config.linear_size, config.linear_size
            )
            self.width_embeddings = torch.nn.Embedding(
                100, config.linear_size, padding_idx=0
            )
        else:
            self.span_linear = torch.nn.Linear(
                hf_config.hidden_size * 2, config.linear_size
            )
            self.width_embeddings = None


        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(config.linear_size * 3, config.linear_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=hf_config.hidden_dropout_prob),
            torch.nn.Linear(config.linear_size, config.linear_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=hf_config.hidden_dropout_prob),
            torch.nn.Linear(config.linear_size, 2),
        )
        
        # self.start_loss_weight = config.start_loss_weight
        # self.end_loss_weight = config.end_loss_weight
        # self.span_loss_weight = config.span_loss_weight
        # self.threshold_loss_weight = config.threshold_loss_weight
        # self.ner_loss_weight = config.ner_loss_weight

        # Initialize weights and apply final processing
        self.post_init()

        self.text_encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
            add_pooling_layer=False,
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self):
        self.text_encoder.gradient_checkpointing_enable()
        self.type_encoder.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        quote_start = None,
        quote_end = None,
        labels = None,
        return_dict: bool = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.hf_config.use_return_dict
        )

        outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        
        # (n_mention_pair*2) x (lm_hidden_size*2) --> n_mention_pair x 2 x hidden_size
        span_output = torch.cat(
            [
                sequence_output[bidx_start, mp_start],
                sequence_output[bidx_end, mp_end],
            ],
            dim=-1,
        )
        if self.width_embeddings is not None:
            
            span_width = mp_end - mp_start
            # seq_length x seq_length x hidden_size
            span_width_embeddings = self.width_embeddings(span_width * (span_width > 0))
            span_output = torch.cat([
                span_output, span_width_embeddings], dim=-1)

        span_output = F.normalize(
            self.dropout(self.span_linear(span_output)), dim=-1
        )

        #### NEGATIVE SAMPLES ####

        #n_clusters, n_neg_samples = neg_bidx_start.size()

        # (n_cluster*n_neg_mention) x lm_hidden_size
        sequence_neg_start = sequence_output[
            neg_bidx_start, neg_start
        ]
        sequence_neg_end = sequence_output[neg_bidx_end, neg_end]

        # (n_cluster*n_neg_mention) x hidden_size
        # sequence_neg_start_output = F.normalize(self.dropout(self.start_linear(sequence_neg_start)), dim=-1)
        # sequence_neg_end_output = F.normalize(self.dropout(self.end_linear(sequence_neg_end)), dim=-1)

        # batch_size x num_types x seq_length
        # start_scores = self.start_logit_scale.exp() * type_start_output.unsqueeze(0) @ sequence_start_output.transpose(1, 2)
        # end_scores = self.end_logit_scale.exp() * type_end_output.unsqueeze(0) @ sequence_end_output.transpose(1, 2)

        #  (n_neg_mention) x (lm_hidden_size*2)
        neg_span_output = torch.cat(
            [
                sequence_neg_start,
                sequence_neg_end,
            ],
            dim=-1,
        )
        if self.width_embeddings is not None:
            
            span_width = neg_end - neg_start
            # seq_length x seq_length x hidden_size
            span_width_embeddings = self.width_embeddings(span_width * (span_width > 0))
            neg_span_output = torch.cat([
                neg_span_output, span_width_embeddings], dim=-1)
            
        #  n_cluster x n_neg_mention x hidden_size
        neg_span_output = F.normalize(
            self.dropout(self.span_linear(neg_span_output)), dim=-1
        )
        print(neg_span_output.size())
        #.view(n_clusters, n_neg_samples, -1)
        # n_mention x n_neg_mention x hidden_size
        # neg_span_output = neg_span_linear_output[k_assignement]

        #### CALCULATING LOSS ####
        span_positive_score = (
            self.span_logit_scale.exp()
            * torch.bmm(
                span_output[:, 0].unsqueeze(1), span_output[:, 1].unsqueeze(2)
            ).squeeze()
        )
        print(span_output[:, 0].unsqueeze(1).size(), neg_span_output.unsqueeze(1).size())
        
        span_negative_score = self.span_logit_scale.exp() * torch.bmm(
            span_output[:, 0].unsqueeze(1),
            neg_span_output.transpose(1,2),
        ).squeeze(1)

        loss = span_positive_score.exp() / (
            (span_negative_score.exp().sum(1) + span_positive_score.exp())
        )
        loss = -torch.log(loss).mean()

        # Predictive Loss

        # [n_mentions x lm_hidden_size * 2]
        span_output = torch.cat(
            [
                sequence_output[gold_bidx_s, gold_s],
                sequence_output[gold_bidx_e, gold_e],
            ],
            dim=1,
        )
        
        if self.width_embeddings is not None:
            
            span_width = gold_e - gold_s
            # seq_length x seq_length x hidden_size
            span_width_embeddings = self.width_embeddings(span_width * (span_width > 0))
            span_output = torch.cat([
                span_output, span_width_embeddings], dim=-1)
            
        # [n_mentions x hidden_size]
        contrastive_repr = F.normalize(self.dropout(self.span_linear(span_output)), dim = -1)

        # [n_mentions, n_antecedents]
        _, top_antecedents = torch.topk(
            contrastive_repr @ contrastive_repr.t(), k=25, dim=1, sorted=False
        )

        predictive_repr = self.dropout(self.span_linear_2(span_output))

        # [n_mentions x hidden_size * 2]
        #mention_repr = torch.cat([contrastive_repr, repr2], dim=-1)
        
        # 
        # [n_mentions x 1]
        #mention_repr = self.predictor(mention_repr)

        # [n_mentionx, n_antecedents]
        index = torch.LongTensor([[i] * 25 for i in range(span_output.size(0))])
        
        # [n_mentions, n_antecedents]
        mention_repr = self.dropout(self.predictor(torch.cat([
            predictive_repr[index],
            predictive_repr[top_antecedents],
            predictive_repr[index] * contrastive_repr[top_antecedents]
        ], dim = -1)).squeeze(2))
        
        # [n_mentions, n_antecedents +1]
        mention_repr = torch.cat([
            torch.zeros(mention_repr.size(0), 1, device=mention_repr.device),
            mention_repr
        ], dim = 1)

        span_idxes = torch.arange(mention_repr.size(0), device=mention_repr.device)
        # [n_mentions, n_mentions]
        ant_offsets_of_spans = span_idxes.view(-1, 1) - span_idxes.view(1, -1)
        # [n_mentions, n_mentions]
        full_ant_mask_of_spans = ant_offsets_of_spans >= 1
        # [n_mentions, n_antecedents]
        full_ant_mask_of_spans = full_ant_mask_of_spans[
            span_idxes.view(-1, 1), top_antecedents
        ]
        
        # [n_mentions, 1]
        non_dummy_span_mask = (gold_cluster_ids != -1).view(-1, 1)

        # [n_mentions, n_antecedents]
        cluster_ant_mask = gold_cluster_ids[top_antecedents]
        cluster_ant_mask += torch.log(full_ant_mask_of_spans.float()).long()
        
        cluster_ant_mask = (gold_cluster_ids.unsqueeze(1) == cluster_ant_mask).long()
        
        # [n_mentions, n_antecedents]
        non_dummy_top_ant_indicators_of_spans = cluster_ant_mask & non_dummy_span_mask
        
        top_ant_indicators_of_spans = torch.cat(
            (
                # [top_cand_num, 1]
                ~non_dummy_top_ant_indicators_of_spans.any(dim=1, keepdim=True),
                # [top_cand_num, pruned_ant_num]
                non_dummy_top_ant_indicators_of_spans
            ), dim=1
        )
        
        # [n_mentions x num_antecedents]
        #mention_repr = (mention_repr @ mention_repr.t())[
        #    span_idxes.view(-1, 1), top_antecedents
        #]

        # print(torch.where(torch.log(cluster_ant_mask.float()).sum(1) == torch.inf))

        #print(
        #    torch.logsumexp(torch.log(top_ant_indicators_of_spans.float()), dim=-1)
        #)

        predictive_loss = -(
            torch.logsumexp(mention_repr + torch.log(top_ant_indicators_of_spans.float()), dim=1)
            - torch.logsumexp(mention_repr, dim=1)
        ).sum()

        # if not return_dict:
        #    output = (start_scores, end_scores, span_scores) + outputs[2:]
        #    return ((total_loss,) + output) if total_loss is not None else output

        #total_loss = 0.5 * loss + 0.5 * predictive_loss
        
        total_loss = loss 
        if not return_dict: 
            return (total_loss, loss, predictive_loss)
        
        return CRCorefModelOutput(
            loss=total_loss,
            contrastive_scores=loss,
            predictive_scores=predictive_loss,
            # start_scores=start_scores,
            # end_scores=end_scores,
            #span_scores=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )