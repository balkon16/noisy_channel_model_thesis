# The current file provides a way to handle creating a bigram language model
# (count based) with multiprocessing tools from Python's standard library.

# The relatively big text file containing bigrams must be divided into smaller
# chunks so that I can use multiprocessing tools. Each partial file will be at
# most 1M lines long.

# I use split_large_file.sh script.

import os
import multiprocessing as mp
import time
from collections import Counter
import pickle
import subprocess

from language_model import apply_word_treatment, \
						   create_two_gram_counts_multiprocessed

# split the large bigram file into smaller ones in order to facilitate
# multiprocessing
subprocess.call("./split_large_file.sh", shell=True)

t0 = time.time()
pool = mp.Pool(processes=6)

# list to store results that will be later merged
results = []

results = [pool.apply_async(create_two_gram_counts_multiprocessed, \
	args=(file_name,)) for file_name in os.listdir('./data/2_gram_partials')]

# define Counter object that sums all the occurences from interim dictionaries
c = Counter()

for res in results:
	c.update(res.get())

bi_gram_counts = dict(c)

with open('./pickles/two_gram_model.p', 'wb') as two_gram_pickle:
    pickle.dump(bi_gram_counts, two_gram_pickle, \
    	protocol=pickle.HIGHEST_PROTOCOL)

t1 = time.time()
print("Time elapsed: ", t1 - t0)
