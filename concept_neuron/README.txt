This folder contains the necessary files to train the logistic regression
classifiers to find concept neurons.

Files / Directories:
- concept_neuron.py: trains one-vs-all logistic regression classifiers to find
                     concept neurons
- concept_neuron_accuracy.py: computes the overall accuracy of all classifiers
                              to get general metrics (accuracy, precision,
                              recall) about the dataset

- process_log_file_to_html.py: converts the log files generate by the
                               classifiers into html tables

- data: holds the data used to train the classifiers

- results: holds the train classifiers for each experiment

How to run:
- Train the classifiers (redirect always the stdout to a log file):
    - python3 concept_neuron.py --help | tee log.txt
