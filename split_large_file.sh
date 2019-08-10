#!/bin/bash

cd ./data/
mkdir ./2_gram_partials/

split -l 1000000 ./2grams --numeric-suffixes ./2_gram_partials/partial_2grams_

echo "Splitting done"
