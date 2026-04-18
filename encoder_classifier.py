#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024
#
# @author: prachi@andrew.cmu.edu, fhammed@andrew.cmu.edu

"""
11-411/611 NLP Assignment 2
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        """
        Positional Encoding module that adds positional information to embeddings.

        Uses sine and cosine functions:
            PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_classes, nhead=2, num_layers=2,
                 dim_feedforward=128, dropout=0.1, max_seq_len=64):
        super(TransformerEncoderClassifier, self).__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
                                   else "cpu")
        print(f"Using device: {self.device}")

        self.d_model = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(self.d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.to(self.device)

    def forward(self, x, attention_mask=None):
        x = x.to(self.device)

        # embed tokens and scale by sqrt(d_model)
        emb = self.embedding(x) * math.sqrt(self.d_model)

        # add positional encoding
        emb = self.positional_encoding(emb)

        # transformer expects (seq, batch, dim)
        emb = emb.transpose(0, 1)

        # padding mask: True where we want to IGNORE (opposite of attention_mask)
        key_padding_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            key_padding_mask = (attention_mask == 0)

        out = self.transformer_encoder(emb, src_key_padding_mask=key_padding_mask)

        # back to (batch, seq, dim)
        out = out.transpose(0, 1)

        # mean pool over non-padded positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)

        return self.fc(self.dropout(pooled))

    def predict(self, x, attention_mask=None):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        return predictions
