#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 12:56:31 2018

@author: ricktjwong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab

glove = vocab.GloVe(name='6B', dim=100)
print('Loaded {} words'.format(len(glove.itos)))


def get_word(word):
    """ 
    Vectors returns the actual vectors.
    To get a word vector get the index to get the vector
    glove.stoi string-to-index returns a dictionary of words to indexes
    """
    return glove.vectors[glove.stoi[word]]


def closest(vec, n=10):
    """
    Find the closest words for a given vector
    """
    # itos index-to-string returns an array of words by index
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in glove.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]


def print_tuples(tuples):
    for tuple in tuples:
        print('(%.4f) %s' % (tuple[1], tuple[0]))


# In the form w1 : w2 :: w3 : ?
def analogy(w1, w2, w3, n=5, filter_given=True):
    print('\n[%s : %s :: %s : ?]' % (w1, w2, w3))
   
    # w2 - w1 + w3 = w4
    closest_words = closest(get_word(w2) - get_word(w1) + get_word(w3))
    
    # Optionally filter out given words
    if filter_given:
        closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]
        
    print_tuples(closest_words[:n])

print_tuples(closest(get_word('king')))

print(analogy('king', 'man', 'queen'))
print(analogy('earth', 'moon', 'sun')) # Interesting failure mode

