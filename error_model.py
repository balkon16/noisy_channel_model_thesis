"""
Error model's (or channel model's) job is to compute the intended word c
given that we have observed the word w.

Generally we don't know what errors are made at what frequency. This can be
estimated by using the Wikipedia revision data.
"""

import pickle
import string
import pandas as pd
import numpy as np
from collections import defaultdict
import operator
import dill # third party library that is used instead of pickle in order to
# load pickled lambda expressions
import time
import re
import sys

from dl2 import get_basic_operations
from create_n_gram_models_with_probs import one_gram_dict_probs


# debugging purposes
import pprint
pp = pprint.PrettyPrinter(indent=4)
############################

# get the set of valid words - according to one-gram model
with open('./pickles/valid_words_set.p', 'rb') as file:
    valid_words = pickle.load(file)

# load dataframes that contain pairs of characters and their count
add_matrix = pd.read_pickle('./pickles/add_matrix_smoothed.p')
del_matrix = pd.read_pickle('./pickles/del_matrix_smoothed.p')
rev_matrix = pd.read_pickle('./pickles/rev_matrix_smoothed.p')
sub_matrix = pd.read_pickle('./pickles/sub_matrix_smoothed.p')
chars_matrix = pd.read_pickle('./pickles/char_matrix_smoothed.p')

with open('./pickles/two_gram_probs.p', 'rb') as file:
    two_gram_probs = pickle.load(file)

def get_candidates(word):
    """
    Get all valid words that are different from `word` by one basic operation
    only.
    """
    latin_letters    = 'abcdefghijklmnoprstuwyz'
    polish_letters = 'ęóąśłźżćń'
    letters = latin_letters + polish_letters
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    # the candidate list usually contains few duplicates so set is used in order
    # to filter them out
    all_candidates = set(deletes + transposes + replaces + inserts)
    try:
        # handle error words that consists of single 'x' or 'q' or 'v' letters
        all_candidates.remove(word)
    except:
        pass

    return all_candidates.intersection(valid_words)

def clean_sentence(sentence):
    """
    Delete characters from the sentence.
    """

    sentence = sentence.replace("\\t", " ")
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

def compute_channel_model_prob(observed, candidate):
    """
    Based on the count of two adjacent characters estimate conditional
    probability P(observed|candidate).
    """

    # get one of the four basic operations that are needed to turn the candidate
    # into the observation
    # nums refer to the index of letters that are vital for the
    # operation
    operation, *nums = get_basic_operations(candidate, observed)[0]
    # choose one of the matrices accordingly
    if operation == 'delete':
        # if the letter of interest is the first one
        if nums[0] == 0:
            row, col = '#', candidate[nums[0]]
        else:
            row, col = candidate[nums[0]-1], candidate[nums[0]]
        # pandas dataframes are dictionaries of columns. In order to access
        # element at row, col you must dataframe[col][row]
        nominator = del_matrix[col][row]
    elif operation == 'insert':
        row, col = candidate[nums[0]], observed[nums[1]]
        nominator = add_matrix[col][row]
    elif operation == 'replace':
        row, col = observed[nums[0]], candidate[nums[0]]
        nominator = sub_matrix[col][row]
    else:
        row, col = candidate[nums[1]], candidate[nums[0]]
        nominator = rev_matrix[col][row]

    # debugging
    try:
        return nominator / chars_matrix[col][row]
    except:
        print("Col: ", col)
        print("Row: ", row)

def compute_language_model_prob(word, antecedent=None, after=None, \
                                    lambda_coef=0.75):

    """
    Given a word compute a (by default) a unigram probability.
    If two `context_words` are provided then a bigram probability is computed.
    If there are two `context_words` then it is assumed that the first one
    is the antecedent and the second one is the word after it.
    """
    # bigram case
    # probability 1 is: P(word|context_words[0])
    # probability 2 is: P(context_words[1]|word)
    # I use Jelinek-Mercer smoothing in order to handle situations in which
    # a bigram never occured in the data that the bigram model is based on.
    # Jelinek-Mercer smoothing uses a lambda coefficient that is effectively
    # a weight that indicates importance of unigram model in bigram
    # probability estimation

    if antecedent and after:
        try:
            bigram_part = lambda_coef*two_gram_probs[antecedent][word]
        except:
            bigram_part = 0

        prob1 = bigram_part + (1-lambda_coef)*one_gram_dict_probs[antecedent]

        try:
            bigram_part = lambda_coef*two_gram_probs[word][after]
        except:
            bigram_part = 0

        prob2 = bigram_part + (1-lambda_coef)*one_gram_dict_probs[word]

        prob = np.exp(np.log(prob1) + np.log(prob2))

    elif antecedent:
        try:
            bigram_part = lambda_coef*two_gram_probs[antecedent][word]
        except:
            bigram_part = 0
        prob =  bigram_part + (1-lambda_coef)*one_gram_dict_probs[antecedent]

    elif after:
        try:
            bigram_part = lambda_coef*two_gram_probs[word][after]
        except:
            bigram_part = 0
        prob =  bigram_part + (1-lambda_coef)*one_gram_dict_probs[word]

    else:
        # unigram case
        prob = one_gram_dict_probs[word]
    return prob

def correct_mistake(sentence, error, use_bigrams=False):
    """
    The assumption is that the error consists of letters only.
    """rror)

    # some errors are listed with punctuation, capital letters
    # if an error consists of banned characters only an exception is raised
    # and handled
    try:
        error = clean_sentence(error)[0]
    except IndexError:
        # you can't generate sensible candidates from an empty string
        return None

    # clean sentence
    clean_tokens = clean_sentence(sentence)

    # get correction candidates
    candidates = get_candidates(error)
    print("Set of candidates: ", candidates)
    # probability for each candidate will be stored in a dictionary indexed
    # with candidates
    cand_probs = defaultdict(float)

    # due to the lack of some correct words in valid_words set
    # e.g. 'charakteryzującym' an exception must be made for empty sets of
    # candidates
    # empty sets corresponds to False
    if not bool(candidates):
        return None

    for cand in candidates:
        # error (channel) model is the same irrespective of the usage of bigrams
        # error is the observation

        error_model_prob = compute_channel_model_prob(error, cand)
        cand_probs[cand] = error_model_prob

    # it may be the case that the sentence such as
    # ! '5: 254, 267 1877 - muszkatałowce'
    # will be transformed into 'muszkatałowce' and no bigram analysis
    # can be produced. This is why the following if block requires that the
    # analysed sentence is at least two-word long

    if use_bigrams and len(clean_tokens) > 1:
        # Depending on the error's position in the sentence one of three
        # situations can occur:
        # 1. The error has an antecedent as well as a word after.
        # 2. The error is the first word in the sentece and has a word after.
        # 3. The error is the last word in the sentence and thus has an
        # antecedent only.
        # Let's say that the error is denoted by w.
        # In the first case a so called full bigram can be computed:
        # P(w|w-1)*P(w+1|w)
        # In the second case we can compute: P(w+1|w).
        # In the third case we can compute: P(w|w-1).

        try:
        # handle imperfect data: error is not in the sentence
            error_location = clean_tokens.index(error)
        except ValueError:
            return None

        for cand, error_model_prob in cand_probs.items():
            if error_location == 0:
                # second case
                language_model_prob =\
                 compute_language_model_prob(cand,
                                after=clean_tokens[error_location+1])

            elif error_location == len(clean_tokens)-1:
                # third case
                language_model_prob =\
                 compute_language_model_prob(cand,
                                    antecedent=clean_tokens[error_location-1])

            else:
                # first case
                language_model_prob = \
                    compute_language_model_prob(cand,
                                    antecedent=clean_tokens[error_location-1],
                                    after=clean_tokens[error_location+1])

            error_model_prob = np.exp(np.log(error_model_prob) \
                                        + np.log(language_model_prob))

    else:
        for cand, error_model_prob in cand_probs.items():
            try:
                language_model_prob = one_gram_dict_probs[cand]
            except NameError:
                pass

            error_model_prob = np.exp(np.log(error_model_prob) \
                                        + np.log(language_model_prob))

    # return the candidate that has the highest probability
    # return max(cand_probs.items(), key=operator.itemgetter(1))[0]
    # debugging:
    most_likely_cand = max(cand_probs.items(), key=operator.itemgetter(1))[0]
    return most_likely_cand

if __name__ == "__main__":
    # import test dataset
    # test dataset is comprised of a dataframe
    with open('./pickles/test_train.dat', 'rb') as file:
        data = pickle.load(file)
        test_df = data[1]
        # gold standard is originally a pandas Series object
        gold_standard = data[3]

    # as is the case with raw errors gold standard entries must be transformed
    # as well
    test_df['gold_standard'] = gold_standard
    test_df['gold_standard'] = test_df['gold_standard'].apply(lambda x: clean_sentence(x)[0])

    print("Started unigram")

    # empty list to hold unigram method predictions
    unigram_case_results = []

    # for loop for the purposes of debugging
    for i in range(test_df.shape[0]):
        sent_with_error = test_df.iloc[i]['text_with_error']
        error = test_df.iloc[i]['error']
        try:
            result = correct_mistake(sent_with_error, error, use_bigrams=False)
        except:
            result = 'other_error'
        unigram_case_results.append(result)

    print("Started bigram")

    bigram_case_results = []

    for i in range(test_df.shape[0]):
        sent_with_error = test_df.iloc[i]['text_with_error']
        error = test_df.iloc[i]['error']
        try:
            # result = correct_mistake(sent_with_error, error, use_bigrams=True)
            # debugging
            result = correct_mistake(sent_with_error, error, use_bigrams=True)
        except:
            result = 'other_error'
        bigram_case_results.append(result)

    test_df['unigram_case'] = unigram_case_results
    test_df['bigram_case'] = bigram_case_results

    file_to_save = 'test_set_with_answers' + '.csv'
    test_df.to_csv(file_to_save)
