// Author: João Carvalho - carvalhj@cs.uni-freiburg.de

This folder contains the code and results developed during the master project, 
developed in the Algorithms and Datastructures chair of the Technical Faculty
at the University of Freiburg.

Generally it contains: 
    - code to train and sample LSTM language models from raw text
    - code to find the sentiment neuron discovered by openAI
    - code to analyse and discover POS tags neurons
    - a webapp to showcase the work done during the project


Files / Directories:

    - Dockerfile:
        - configures a docker environment to launch the webapp

    - byte_LSTM:
        - python scripts to train and sample byte-language models using
          tensorflow and riseml

    - byte_LSTM_trained_models:
        - language models trained with the amazon product reviews and
          wikitext-103 datasets

    - concept_neuron:
        - python scripts to analyse the concept neurons of the trained
          language models

    - preprocess_data:
        - python scripts to preprocess the wikitext and amazon reviews datasets

    - sent_neuron:
        - python scripts to analyse the sentiment neuron from openAI

    - www:
        - webapp with the information gathered along the project, along with
          the results obtained

    - setup.py:
        - installs the byte_LSTM package to load LSTM models

    - setup_directory.sh:
        - configures a python virtual environment with the packages needed to run
          localy the experiments and the webapp

    - requirements.txt:
        - list of python dependencies


How to run:

    - To launch the webapp run: 
        - svn co https://ad-svn.informatik.uni-freiburg.de/student-projects/joao-carvalho
        - cd joao-carvalho
        - docker build -t project .
        - docker run -it -p 5000:5000 -v /nfs/students/joao-carvalho:/extern/data project
