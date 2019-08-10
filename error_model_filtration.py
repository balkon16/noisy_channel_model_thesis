"""
One-time operation: based on the list provided by SJP
(https://sjp.pl/slownik/growy/) create a set of valid Polish words.
"""

import pickle

valid_words = set()
with open('slowa.txt', 'r') as file:
    for line in file:
        # delete trailing whitespaces
        line = line.rstrip()
        valid_words.add(line)
        
with open('./pickles/valid_words_set.p', 'wb') as file:
    pickle.dump(valid_words, file, protocol=pickle.HIGHEST_PROTOCOL)
