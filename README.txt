// Author: João Carvalho - carvalhj@cs.uni-freiburg.de

This folder contains the code and results developed during the master project.

It includes: 
- code to train and sample LSTM language models from raw text;
- code to find the sentiment neuron discovered by openAI;
- code to analyse and discover POS tags neurons.


Files / Directories:

- Dockerfile:
    - configures a docker environment to launch the webapp

- byte_LSTM:
    - python scripts to train and sample byte-language models using
      tensorflow and the riseml tool

- byte_LSTM_trained_models:
    - language models trained with the amazon product reviews and
      wikitext-103 datasets

- concept_neuron:
    - python scripts to analyse the concept neurons of the trained
      language models

- data:
    - amazon product reviews and the wikitext-103 datasets
    - the amazon product reviews raw data can be obtained from
      http://jmcauley.ucsd.edu/data/amazon/links.html
    - the wikitext-103 dataset can be obtained from
      https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

- preprocess_data:
    - python scripts to preprocess the datasets

- sent_neuron:
    - python scripts to analyse the sentiment neuron from openAI

- www:
    - webapp with the information gathered along the project, along with
      the results obtained

- setup_directory.sh:
    - configures a python virtual environment with the packages needed to run
      the experiments and the webapp

- requirements.txt:
    - python dependencies needed

How to run:
    - To launch the webapp run: 
        - svn co https://ad-svn.informatik.uni-freiburg.de/student-[projects|theses]/<firstname>-<lastname>
        - cd <firstname>-<lastname>
        - docker build -t <name> .
        - docker run -it -p 5000:5000 -v /nfs/students/<firstname>-<lastname>:/extern/data <name>

