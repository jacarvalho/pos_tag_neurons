"""
Copyright 2018
Authors: João Carvalho <carvalhj@cs.uni-freiburg.de>

Split the Amazon Reviews Dataset into shards and preprocess each review.
"""
import os
import gzip
from tqdm import tqdm


# The dataset can be obtained from
# http://jmcauley.ucsd.edu/data/amazon/links.html
# Please, provide absolute paths
data_path = '/local/hdd/exports/data/shared/amazon_reviews/ \
             aggressive_dedup.json.gz'
save_path = '/local/hdd/exports/data/carvalhj/amazon_reviews/'


# Generator function to read a large .gz file
def parse(path):
    with gzip.open(path, 'r') as f:
        for line in f:
            yield eval(line)


gen = parse(data_path)

num_shards = 1000
reviews_per_shard = 82830  # There are 82.83 million reviews

if not os.path.isdir(save_path):
    os.makedirs(save_path)

for i in tqdm(range(num_shards)):
    # Make a folder for each shard of reviews
    shard_folder = os.path.join(save_path, str(i))
    os.makedirs(shard_folder)

    # Preprocess reviews
    # Open text as a binary file to write encoded utf-8 strings
    with open(os.path.join(shard_folder, 'input.txt'), 'ab') as f:
        for _ in range(reviews_per_shard):
            element = next(gen)
            review = element['reviewText'].replace('\n', ' ').strip()
            review = '\n ' + review + '\n'
            f.write(review.encode('utf-8'))
