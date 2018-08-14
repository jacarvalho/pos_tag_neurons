This folder contains the necessary files to replicate the sentiment neuron
paper by openAI. It trains and evaluates a logistic regression classifier.

Files / Directories:
- sst_binary.py: the replication of the openAI experiment
- utlis_sst.py: utilities to prepare data for training and the training of the
                classifier

- data_sst: holds the sst binary dataset

How to run:
- To replicate the experiment of openAI run:
    - python3 sst_binary.py (make sure to set save_dir to the folder with the
                             previsouly trained language model)
