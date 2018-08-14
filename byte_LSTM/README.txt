This folder contains the necessary files to train and sample a LSTM language
model using tensorflow.

Files / Directories:
- requirements.txt: list of PYTHON packages needed for riseml
- riseml_*: riseml training file
- start_train_*: bash script to train models in shards

- model.py: describes a LSTM network in tensorflow
- sample.py: generates text from a pre-trained LSTM model
- train.py: trains a LSTM language model
- utlis.py: utilities to prepare data for training
- plot_loss_accuracy.py: plots losses and accuracies of trained models

- clear_directories.sh: utility to clear clear directories 

- tests: holds the tests of some utility functions

How to run:
- To train a language model run either:
    - python3 train.py --help (to train it in your machine)
    - riseml train -f riseml_* (to train with riseml)

- To sample a languge model run:
    - python3 sample.py --help
