This folder contains scripts to split data from the Amazon Reviews and
Wikitext-103 datasets into shards.

The files do not have any CLI. Open them and change the top lines to indicate
the paths where the original data is located.

Files / Directories:
    - shard_data_amazon_reviews.py: splits and preprocesses amazon reviews
    - shard_data_wikitext.py: splits and preprocesses articles from wikitext-103

How to run:
    - In each *.py script set the options in the global variables inside the
      script, and simply run as python3 *.py
