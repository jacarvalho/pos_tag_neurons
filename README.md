# Concept Neurons

This project studies the capabilities of LSTM networks to train language models from byte sequences, and investigates the ability of the hidden representation to encode information on Part-Of-Speech (POS) tags. 

Check the full report in [http://ad-publications.informatik.uni-freiburg.de/student-projects/concept-neurons](http://ad-publications.informatik.uni-freiburg.de/student-projects/concept-neurons). Run the code as instructed below to get a fully interactive webpage.


---

### Contents:

- Code to train and sample LSTM language models from raw text
- Code to find the [sentiment neuron](https://blog.openai.com/unsupervised-sentiment-neuron/) discovered by openAI 
- Code to analyse and discover Part-of-Speech (POS) tags neurons
- A webapp to experiment with the trained language models



### Files / Directories:

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

---

### How to run:
To launch the webapp run: 

    - git clone https://github.com/jacarvalho/concept_neurons
    - cd concept_neurons
    - docker build -t project .
    - docker run -it -p 5000:5000 -v /abs/path/to/directory/:/extern/data project
