import datetime
import functools
import math
import os
import re

import nltk
import torch
from transformers import BartTokenizer, T5Tokenizer
from torch.nn import functional as F
#from .data_utils import getClassLabels

nn = torch.nn
def getClassLabels(nb_classes):
    # The class label token is represented as #C{chr(i+97).upper()}
    classes = []
    for i in range(nb_classes):
        cl = '#C'+chr(i+97).upper()
        classes.append(cl)
    return classes
classes_tokens = getClassLabels(7)


def setupTokenizer(modelbase):
    if 't5' in modelbase:
        tokenizer_ = T5Tokenizer.from_pretrained(modelbase)
    elif 'bart' in modelbase:
        tokenizer_ = BartTokenizer.from_pretrained(modelbase)
    classification_metrics = ['F1-Score',
                              'F2-Score', 'F1-score', 'F2-score', 'F1score', 'F2score', 'G-Mean']
    rates_vocabulary = rates_vocabulary = [
        'VALUE_HIGH', 'VALUE_MODERATE', 'VALUE_LOW']  # ['HIGH','MODERATE','LOW']
    # classification_metrics +
    # ,'<acc_diff>',
    additional_vocab = rates_vocabulary + classes_tokens+['also_known_as', 'ml_task','is_imbalanced','is_balanced','data_dist', '<|IMBALANCED|>', '<|BALANCED|>', 'class_labels', 'metric_value', 'metric_rate', 'dataset_attributes', '<|majority_dist|>',
                                                          '<|minority_dist|>', '<rec_diff>', '<preci_diff>', '<acc_diff>']+classification_metrics
    if 't5' in modelbase:
        special_tokens = ['<|>', '&&', '<TaskDec>',
                          '<MetricsInfo>', '<|table2text|>']
        tokenizer_.add_special_tokens({'sep_token': '<|section-sep|>',
                                       'additional_special_tokens': special_tokens})
    elif 'bart' in modelbase:
        special_tokens = ['<|>', '&&', '<TaskDec>',
                          '<MetricsInfo>', '<|table2text|>', '<|section-sep|>']
        tokenizer_.add_special_tokens(
            {'additional_special_tokens': special_tokens})

    tokenizer_.add_tokens(additional_vocab)
    return tokenizer_


class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        k_t = self.key(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out) + x

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel