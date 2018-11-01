"""
2018, University of Freiburg.
Joao Carvalho <carvalhj@cs.uni-freiburg.de>
"""
import os
import argparse
import pickle
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score
from concept_neuron import split_train_valid_test, process_sentence_pos_tags
from concept_neuron import print_pos_tag_statistics, compute_LSTM_states


# hidden_states or cell_states of LSTMs
state_type = 'cell_states'

# List of concepts to analyse - Upenn POS tags
# http://www.nltk.org/api/nltk.tag.html
# To find the available POS tags:
#   import nltk.help; nltk.help.upenn_tagset()
concepts = ['(', ')', ',', '.', 'CC', 'CD', 'DT', 'IN', 'JJ', 'MD',
            'NN', 'NNP', 'PRP', 'RB', 'TO', 'VB']
concepts.extend(['SPACE', 'OTHER'])


def concept_neurons_accuracy(args):
    """
    Computes the accuracy for various logistic regression classifiers
    for different POS tags, as a multiclass classifier.

    Args:
        args (argparse): arguments.

    Returns:
        None.
    """
    # Directory with LSTM model.
    save_dir = args.save_dir

    # Folder to save results.
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    results_dir = args.results_dir

    # Data to analyse.
    input_file = args.data_file

    # Get training data, tokenize and POS tag sentences.
    # X holds the sentences (word1, word2, ...)
    # Y holds the corresponding ((word1, tag1), (word2, tags), ...)
    X, Y = process_sentence_pos_tags(input_file, args.group_tags)

    # Set the concepts to the whole set if no grouping is required.
    unique_tags, counts = np.unique([y[1] for sublist in Y for y in sublist],
                                    return_counts=True)
    if not args.group_tags:
        global concepts
        concepts = unique_tags

    # Print some statistics about the initial distribution of POS tags.
    print_pos_tag_statistics(unique_tags, counts)

    # Computes the LSTM state for each byte in X.
    X_t, X_t_pos_tags = compute_LSTM_states(save_dir, X, Y)

    # Compute the overall metrics for the logistic regression classifiers.
    print('\n-----> Test results')
    classifiers_id = ['all', 'top1', 'top2', 'top3']
    for classifier_id in classifiers_id:
        print('\n- {}'.format(classifier_id))
        concept_classifiers = []
        predicted_probs = []
        classes = []
        for concept in concepts:
            lr_file = os.path.join(
                results_dir, 'log_reg_model_' + concept +
                '_' + classifier_id + '.sav')
            if not os.path.exists(lr_file):
                continue
            concept_classifiers.append(concept)
            lr_model = pickle.load(open(lr_file, 'rb'))
            classes.append(lr_model.classes_[0])

            # Largest coefficients
            lr_file_all = os.path.join(
                results_dir, 'log_reg_model_' + concept + '_all.sav')
            coef_sorted = np.argsort(-np.abs(np.squeeze(
                pickle.load(open(lr_file_all, 'rb')).coef_)))

            x = re.search(r'^top(?P<k>\d)$', classifier_id)
            if x is None:  # all weights
                X_t_ = X_t
            else:  # top k weights
                k = int(x.group('k'))
                X_t_ = [x[coef_sorted[0:k]] for x in X_t]

            trX, vaX, teX, trY, vaY, teY = split_train_valid_test(X_t_,
                                                                  X_t_pos_tags)
            predicted_probs.append(lr_model.predict_proba(teX)[:, 0])

        # Find the class with largest predicted probability.
        concept_classifiers = np.array(concept_classifiers)
        predicted_probs = np.array(predicted_probs)
        max_prob_ind = np.argmax(predicted_probs, axis=0)
        pred_classes = concept_classifiers[max_prob_ind].tolist()

        y_true, y_pred = teY, pred_classes
        print('Test accuracy:  {:.3f}'.format(accuracy_score(y_true, y_pred)))
        print('Test precision: {:.3f}'.format(
            precision_score(y_true, y_pred, average='weighted')))
        print('Test recall: {:.3f}'.format(
            recall_score(y_true, y_pred, average='weighted')))


if __name__ == '__main__':
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--save_dir', type=str,
        default='../byte_LSTM_trained_models/wikitext/save/95/',
        help='directory containing LSTM-model')
    parser.add_argument('--data_file', type=str, default=None,
                        help="""file to use as input to the classifier.
                                If no file is provided, the
                                nltk.corpus.treebank is used
                        """)
    parser.add_argument('--results_dir', type=str, default='results',
                        help='directory with saved classifiers')
    parser.add_argument('--group_tags', action='store_true',
                        help="""group all VB* tags into VB;
                                JJ* into JJ;
                                NN* into NN;
                                NNP* into NNP;
                                RB* into RB.
                        """)

    args = parser.parse_args()
    concept_neurons_accuracy(args)
