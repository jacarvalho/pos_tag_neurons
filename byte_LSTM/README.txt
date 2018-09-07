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
    - To train a language model on your machine:
        - Consider the following input parameters:
            - data: a text file named input in ./data/0/input.txt (0 is needed, 
                    because the data can be split into different shards)
            - lstm size: 128 units
            - batch size: 128
            - sequence length: 128
            - epochs: 10
            - constant learning rate: 0.001
        - Run:
            - python3 train.py --option=train  --data_dir=./data/
                               --rnn_size=128  --batch_size=128
                               --seq_length=128 --num_epochs=1
                               --lr_init=0.001

    - To train a language model using riseml:    
        - Set in riseml_* and in start_train_* your own parameters
        - Run:        
            - riseml train -f riseml_*
        
    - To sample a previously trained languge model:
        - Assume the language model was saved in ./save/0 and we want to
          generate 100 bytes
        - Run:
            - python3 sample.py --save_dir=./save/0 --n=100
