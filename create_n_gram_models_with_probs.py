"""
The following script provides importer scripts with a one-gram probability-
based model that returns a default value when a word (key) is absent in the
dictionary.
"""

import pickle
from collections import defaultdict

with open('./pickles/one_gram_model.p', 'rb') as file:
    one_gram_dict = pickle.load(file)

# use Laplace smoothing in order to avoid zero probabilities

no_of_words = len(one_gram_dict.keys())
basic_denominator = sum(one_gram_dict.values()) #denominator before smoothing
#being applied

default_value = float(1/(no_of_words+basic_denominator))

one_gram_dict_probs = defaultdict(lambda: default_value)

for word, count in one_gram_dict.items():
    one_gram_dict_probs[word] = (count+1) / (basic_denominator+no_of_words)
