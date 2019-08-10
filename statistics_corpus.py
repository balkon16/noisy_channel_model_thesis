import os
from collections import defaultdict
import time
import pickle

from plewic_handler import *

start_time = time.time()
print("Script started")

saving_directory = os.path.join(os.getcwd(), 'pickles')

plewic_directory = os.path.join(os.getcwd(), 'plewic-yaml')

os.chdir(plewic_directory)

# statistics
stats = {}
stats['valid_sentence'] = 0
stats['total_sentences'] = 0
# defaultdict(dict) makes key-value assignment easier (handling KeyError)
# stats['categories'] = defaultdict(dict)
# stats['type'] = defaultdict(dict)
stats['categories'] = defaultdict(int)
stats['type'] = defaultdict(int)

# the list will contain key-value pairs {"filename1": [revision1, revision2, ...],
#                                       "filename2": [revision1, revision2, ...], ...}
files_as_lists = {}


# iterate over files in the ./plewic directory

for plewic_filename in os.listdir(plewic_directory):
    # consider only those files that start with 'plewic' and contain the content
    # we are interested in
    if 'plewic' not in plewic_filename:
        continue

    # the following list contains the file translated into a list of dictionaries
    # one dictionary refers to one revision
    file_list = parse_ruby_yaml_to_python_dict(plewic_filename)

    # add revision list to the dictionary
    files_as_lists[plewic_filename] = file_list

    for revision in file_list:
        # errors
        try:
            for error in revision['errors']:
                # dictionary of attribute, value pairs
                attributes_dict = error['attributes']

                try:
                    stats['categories'][attributes_dict['category']] += 1
                except KeyError:
                    pass

                try:
                    stats['type'][attributes_dict['type']] += 1
                except KeyError:
                    pass

            stats['total_sentences'] += 1
            if revision['valid_sentence']:
                stats['valid_sentence'] += 1
        except KeyError:
            pass

# pp.pprint(stats)

# save to pickles
pickle.dump(stats, open(os.path.join(saving_directory, "stats_corpus.p"), "wb"))
pickle.dump(files_as_lists, open(os.path.join(saving_directory, "plewic_files_as_lists.p"), "wb"))

print("--- %s seconds ---" % (time.time() - start_time))
