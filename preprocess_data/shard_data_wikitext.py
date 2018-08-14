"""
Copyright 2018
Authors: João Carvalho <carvalhj@cs.uni-freiburg.de>

Split the Wikitext Dataset (raw data) into shards and preprocess.
"""
import os
import re


# The dataset can be obtained from
# https://einstein.ai/research/ \
# the-wikitext-long-term-dependency-language-modeling-dataset
# Please, provide absolute paths
train_path = '/local/hdd/exports/data/carvalhj/wikitext/source/ \
              wikitext-103-raw/wiki.train.raw'
valid_path = '/local/hdd/exports/data/carvalhj/wikitext/source/ \
              wikitext-103-raw/wiki.valid.raw'
test_path = '/local/hdd/exports/data/carvalhj/wikitext/source/ \
             wikitext-103-raw/wiki.test.raw'

save_path = '/local/hdd/exports/data/carvalhj/wikitext/shard/'


# Generator function to read a large file
def parse(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            yield eval(line)


# There are approximately 450 000 written lines in wikitext, and we want
def shard_file(data_path, start_shard, articles_per_shard=9000):

    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f_read:
        r = re.compile("(( )*\\n$)|(( = ).)")
        i = 0
        folder = start_shard
        for line in f_read:
            if not re.match(r, line):
                if i == 0:
                    # Make a folder for each shard
                    shard_folder = os.path.join(save_path, str(folder))
                    os.makedirs(shard_folder)
                    folder += 1
                    f_write = open(os.path.join(shard_folder, 'input.txt'),
                                   'ab')

                line = '\n ' + line
                f_write.write(line.encode('utf-8', errors='ignore'))
                i += 1
                if i >= articles_per_shard:
                    i = 0
    return folder


folder = shard_file(train_path, 0)
folder = shard_file(valid_path, folder)
folder = shard_file(test_path, folder)
