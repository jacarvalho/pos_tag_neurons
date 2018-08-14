import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from byte_LSTM.model import Model
import argparse
import nltk
import pickle
import tensorflow as tf
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def concept_neurons_accuracy(args):
    """
    Computes the accuracy for various logistic regression classifiers
    for different POS tags, as a multiclass classifier.

    Args:
        args (argparse): arguments.

    Returns:
        None.
    """
    state_type = 'cell_states'  # hidden_states or cell_states of LSTMs

    # Directory with LSTM model.
    save_dir = args.save_dir

    # Folder to save results.
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    results_dir = args.results_dir

    # Data to analyse.
    input_file = args.data_file

    # List of concepts to analyse - Upenn POS tags
    # http://www.nltk.org/api/nltk.tag.html
    # To find the available POS tags:
    #   import nltk.help; nltk.help.upenn_tagset()
    concepts = ['(', ')', ',', '.', 'CC', 'CD', 'DT', 'IN', 'JJ', 'MD',
                'NN', 'NNP', 'PRP', 'RB', 'TO', 'VB']
    concepts.extend(['SPACE', 'OTHER'])

    def split_train_valid_test(X, Y=None, test_size=0.1, valid_size=0.1):
        """
        Splits X and Y into training, validation and testing subsets.

        Args:
            X, Y (ndarray): input and label datasets
            test_size (float): test dataset size
            valid_size (float): validation dataset size

        Returns:
            trX, vaX, teX, trY, vaY, teY (ndarray): the splitted dataset
        """
        if Y is not None:
            trX, teX, trY, teY = train_test_split(
                X, Y, test_size=test_size, random_state=1)
            trX, vaX, trY, vaY = train_test_split(
                trX, trY, test_size=valid_size, random_state=1)
            return trX, vaX, teX, trY, vaY, teY
        else:
            trX, teX = train_test_split(
                X, test_size=test_size, random_state=1)
            trX, vaX = train_test_split(
                trX, test_size=valid_size, random_state=1)
            return trX, vaX, teX

    """
    Get training data, tokenize and POS tag sentences.
    Use the given input file or the corpus from nltk treebank - a sample of
    tagged sentences from the WSJ
    """
    print('Reading file and POS tagging...')
    if input_file is not None:
        f = open(input_file, 'r', encoding='utf-8', errors='ignore')
        sentences = nltk.sent_tokenize(f.read())
        sentence_tag_tokens = [nltk.pos_tag(nltk.word_tokenize(
            sentence, language='english'), lang='eng')
            for sentence in sentences]
    else:
        sentence_tag_tokens = nltk.corpus.treebank.tagged_sents()[0:20]

    sentences = []
    tags = []
    for pos_tags in sentence_tag_tokens:
        sentence_tmp = ''
        pos_tags_tmp = []
        for word, tag in pos_tags:
            sentence_tmp += word + ' '
            # Group tags
            if args.group_tags:
                # Preprocess tags
                if re.match('VB.*$', tag):  # Group all verbs
                    tag = 'VB'
                elif re.match('JJ.*$', tag):  # Group all adjectives
                    tag = 'JJ'
                elif re.match('NN$|NNS$', tag):  # Group all nouns
                    tag = 'NN'
                elif re.match('NNP$|NNPS$', tag):  # Group all proper nouns
                    tag = 'NNP'
                elif re.match('RB.*$', tag):  # Group all adverbs
                    tag = 'RB'

                if tag in concepts:
                    pass
                else:
                    tag = 'OTHER'
            pos_tags_tmp.append((word, tag))
            pos_tags_tmp.append((' ', 'SPACE'))
        sentences.append(sentence_tmp)
        tags.append(pos_tags_tmp)
    print('...completed.')

    # X holds the sentences (word1, word2, ...)
    # Y holds the corresponding ((word1, tag1), (word2, tags), ...)
    X, Y = sentences, tags

    # Some statistics about the distribution of POS tags
    print('\nPOS tag distribution')
    unique_tags, counts = np.unique([y[1] for sublist in Y for y in sublist],
                                    return_counts=True)
    # Set the concepts to the whole set if no grouping is required.
    if not args.group_tags:
        concepts = unique_tags
    count_sort_ind = np.argsort(-counts)
    print('Identified pairs (word, tag): {}'.format(np.sum(counts)))
    for tag, count in zip(unique_tags[count_sort_ind], counts[count_sort_ind]):
        print('{}  \t-  \t{}  \t-  \t{:.3f}'.format(tag, count,
                                                    count/sum(counts)))
    print()

    """
    Build trained model and compute the state of each word.
    """
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        model_saved_args = pickle.load(f)

    tf.reset_default_graph()
    model = Model(model_saved_args, sampling=True)

    with tf.Session() as sess:
        # Restore the saved session.
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if not ckpt:
            print('Unable to find checkpoint file.')
            sys.exit()
        model_path = os.path.join(
            save_dir, ckpt.model_checkpoint_path.split('/')[-1])
        saver.restore(sess, model_path)

        # X transformed
        X_t = np.array(model.transform(sess, X)[state_type])

    # Associate each byte with a POS tag
    # X_t_pos_tags holds the POS tag for each cell state previously computed
    Y = [item for sublist in Y for item in sublist]
    X_t = [item for sublist in X_t for item in sublist]
    X_t_pos_tags = []
    for word_tag in Y:
        word, tag = word_tag[0], word_tag[1]
        n_bytes = len(word.encode('utf-8'))
        for _ in range(n_bytes):
            X_t_pos_tags.append(tag)

    assert len(X_t) == len(X_t_pos_tags), \
        "Number of processed bytes and POS tags do not match."

    print('Identified pairs (bytes, tag): {}'.format(len(X_t)))
    unique, counts = np.unique(X_t_pos_tags, return_counts=True)
    count_sort_ind = np.argsort(-counts)
    for tag, count in zip(unique[count_sort_ind], counts[count_sort_ind]):
        print('{}  \t-  \t{}  \t-  \t{:.3f}'.format(tag, count,
                                                    count/sum(counts)))

    """
    Compute the overall metrics for the logistic regression classifiers
    """
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

        # Find the class with largest predicted probability
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
