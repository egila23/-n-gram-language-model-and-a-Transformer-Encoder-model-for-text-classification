#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024
#
# @author: Ezra Fu <erzhengf@andrew.cmu.edu>
# based on work by
# Ishita <igoyal@andrew.cmu.edu>
# Suyash <schavan@andrew.cmu.edu>
# Abhishek <asrivas4@andrew.cmu.edu>

"""
11-411/611 NLP Assignment 2
"""

from utils import *
from collections import Counter
from itertools import product
import argparse
import random
import math


def get_ngrams(list_of_words, n):
    """
    Returns a list of n-grams for a list of words.
    Args
    ----
    list_of_words: List[str]
        List of already preprocessed and flattened (1D) list of tokens
    n: int
        n-gram order e.g. 1, 2, 3

    Returns:
        n_grams: List[Tuple]
            Returns a list containing n-gram tuples
    """
    return [tuple(list_of_words[i:i+n]) for i in range(len(list_of_words) - n + 1)]


class NGramLanguageModel():
    def __init__(self, n, train_data, alpha=1):
        """
        Language model class.

        Args
        ____
        n: int
            n-gram order
        train_data: List[List]
            already preprocessed unflattened list of sentences
        alpha: float
            Smoothing parameter

        Attributes
        __________
        self.tokens: list of individual tokens in training corpus
        self.vocab: vocabulary dict with counts
        self.model: n-gram dict with probabilities
        self.n_grams_counts: frequency of each ngram tuple
        self.prefix_counts: frequency of each (n-1)-gram tuple
        """
        self.n = n
        self.alpha = alpha
        self.train_data = train_data
        self.tokens = flatten(train_data)
        self.vocab = Counter(self.tokens)
        self.n_grams_counts = {}
        self.prefix_counts = {}
        self.model = self.build()

    def build(self):
        """
        Build the n-gram model with smoothed probabilities.
        """
        self.n_grams_counts = Counter(get_ngrams(self.tokens, self.n))

        if self.n > 1:
            # get prefix counts by summing over n-gram counts grouped by their (n-1) prefix
            self.prefix_counts = Counter()
            for ngram, cnt in self.n_grams_counts.items():
                self.prefix_counts[ngram[:-1]] += cnt

        return {ngram: self.get_smooth_probabilities(ngram) for ngram in self.n_grams_counts}

    def get_smooth_probabilities(self, ngrams):
        """
        Returns smoothed probability using Laplace Smoothing.
        """
        vocab_size = len(self.vocab)
        count = self.n_grams_counts.get(ngrams, 0)

        if self.n == 1:
            # for unigrams, denominator is total token count
            total = len(self.tokens)
            return (count + self.alpha) / (total + self.alpha * vocab_size)
        else:
            prefix_count = self.prefix_counts.get(ngrams[:-1], 0)
            return (count + self.alpha) / (prefix_count + self.alpha * vocab_size)

    def get_prob(self, ngram):
        """
        Returns probability of the n-gram. Lazy smoothing:
        if the ngram wasn't seen in training, compute and cache it now.
        """
        if ngram not in self.model:
            self.model[ngram] = self.get_smooth_probabilities(ngram)
        return self.model[ngram]

    def perplexity(self, test_data):
        """
        Returns perplexity on the test data.
        """
        test_tokens = flatten(test_data)
        test_ngrams = get_ngrams(test_tokens, self.n)
        N = len(test_tokens)
        log_prob_sum = sum(math.log(self.get_prob(ng)) for ng in test_ngrams)
        return math.exp(-log_prob_sum / N)
