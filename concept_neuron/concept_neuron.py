"""
2018, University of Freiburg.
Joao Carvalho <carvalhj@cs.uni-freiburg.de>
"""
import sys
import os
from byte_LSTM.model import Model
import argparse
import nltk
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


# hidden_states or cell_states of LSTMs
state_type = 'cell_states'

# List of concepts to analyse - Upenn POS tags
# http://www.nltk.org/api/nltk.tag.html
# To find the available POS tags:
#   import nltk.help; nltk.help.upenn_tagset()
concepts = ['(', ')', ',', '.', 'CC', 'CD', 'DT', 'IN', 'JJ', 'MD',
            'NN', 'NNP', 'PRP', 'RB', 'TO', 'VB']
concepts.extend(['SPACE', 'OTHER'])


def train_lrc(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
              C=2**np.arange(-4, 1).astype(np.float), seed=42):
    """
    Train a logistic regression classifier.

    Args:
        trX, trY, vaX, vaY, teX, teY (ndarray): training, validation,
                                                testing datasets
        penalty (str): regularization penalty
        C (ndarray): regularization coefficients
        seed (int): seed for the random generator to shuffle the data

    Returns:
        model (sklearn.linear_model.LogisticRegression): trained classifier

    """
    scores = []
    for i, c in tqdm(enumerate(C)):
        model = LogisticRegression(C=c, penalty=penalty,
                                   random_state=seed+i, tol=0.0001)
        model.fit(trX, trY)
        score = model.score(vaX, vaY)
        scores.append(score)
    c = C[np.argmax(scores)]
    model = LogisticRegression(C=c, penalty=penalty,
                               random_state=seed+len(C), tol=0.0001)
    model.fit(trX, trY)
    return model


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


def process_sentence_pos_tags(input_file, group_tags):
    """
    Get training data, tokenize and POS tag sentences from the input_file.
    Use the given input file or the corpus from nltk treebank - a sample of
    tagged sentences from the WSJ.

    Args:
        input_file (str): the input file path to get data from.
        group_tags (boolean): if True group related POS tags into one.

    Return:
        sentences (list): list of tagged sentences.
        tags (list): (word, POS tag) tuple for each tagged word in a sentence.
    """

    print('Reading file and POS tagging...')
    if input_file is not None:
        f = open(input_file, 'r', encoding='utf-8', errors='ignore')
        sentences = nltk.sent_tokenize(f.read())
        sentence_tag_tokens = [nltk.pos_tag(nltk.word_tokenize(
            sentence, language='english'), lang='eng')
            for sentence in sentences]
    else:
        sentence_tag_tokens = nltk.corpus.treebank.tagged_sents()[0:1000]

    sentences = []
    tags = []
    for pos_tags in sentence_tag_tokens:
        sentence_tmp = ''
        pos_tags_tmp = []
        for word, tag in pos_tags:
            sentence_tmp += word + ' '
            # Group tags
            if group_tags:
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
    return sentences, tags


def print_pos_tag_statistics(unique_tags, counts):
    """
    Print some statistics about the distribution of POS tags.

    Args:
        unique_tags (list): list of uniquely identified tags.
        counts (list): list of the number of occurences for each unique tag.
    """
    print('\nPOS tag distribution')
    count_sort_ind = np.argsort(-counts)
    print('Identified pairs (word, tag): {}'.format(np.sum(counts)))
    for tag, count in zip(unique_tags[count_sort_ind], counts[count_sort_ind]):
        print('{}  \t-  \t{}  \t-  \t{:.3f}'.format(tag, count,
                                                    count/sum(counts)))
    print()


def compute_LSTM_states(save_dir, X, Y):
    """
    Computes the LSTM state of each sentence/word in X and associates each
    byte with a corresponding POS tag.
    Prints statistics of produced (byte, tag) tuples.

    Args:
        X (list): list of sentences to transform through the LSTM network.
        Y (list): list of (word, tag) tuples from X.

    Returns:
        X_t (list): list of ndarray with cell states for each byte of X.
        X_t_pos_tags (list): list of POS tag for each byte of X.
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

    return X_t, X_t_pos_tags


def compute_lrc(concept, X_t_, X_t_pos_tags_, trX, vaX, teX, trY, vaY, teY,
                results_dir):
    """
    Trains and saves a logistic regression classifier for the given concept.
    Outputs the metrics of the trained logistic regression classifier.
    Saves plots to results_dir with weights of the logistic regression
    classifier.

    Args:
        concept (str): POS tag corresponding to the logistic regression
                       classifier.
        X_t_ (list): list of ndarray with LSTM states.
        X_t_pos_tags_ (list): list of POS tags for each byte in X_t_.
        trX (ndarray): array with training input data.
        vaX (ndarray): array with validation input data.
        teX (ndarray): array with test input data.
        trY (ndarray): array with training target data.
        vaY (ndarray): array with validation target data.
        teY (ndarray): array with test target data.
        results_dir (str): the path to save results to.

    Returns:
        lr_model (LogisticRegression): a sklearn logistic regression model for
                                       the concept.

    """
    # Print some statistics about the distribution.
    print('\nDataset statistics')
    dataset = {'trY': trY, 'vaY': vaY, 'teY': teY}
    for item in dataset:
        print('-- Data: {}'.format(item))
        unique, counts = np.unique(dataset[item], return_counts=True)
        count_sort_ind = np.argsort(-counts)
        for tag, count in zip(unique[count_sort_ind],
                              counts[count_sort_ind]):
            print('{}  \t-  \t{}  \t-  \t{:.3f}'.format(tag, count,
                                                        count/sum(counts)))

    print('\n--- ALL dimensions')
    # Train a Logistic Regression Classifier on top of all dimensions.
    print('\nTraining a Logistic Regression Classifier...')
    lr_model = train_lrc(trX, trY, vaX, vaY, teX, teY)
    print('...completed.')

    lr_file = results_dir + '/log_reg_model_' + concept + '_all' + '.sav'
    pickle.dump(lr_model, open(lr_file, 'wb'))

    print('\n--> Logistic Regression results.')
    print('Train accuracy: {:.3f}'.format(lr_model.score(trX, trY)))
    print('Valid accuracy: {:.3f}'.format(lr_model.score(vaX, vaY)))
    print('Test  accuracy: {:.3f}'.format(lr_model.score(teX, teY)))
    print('Regularization coef: {:.3f}\n'.format(lr_model.C))

    y_true, y_pred = teY, lr_model.predict(teX)
    print('Test accuracy:  {:.3f}'.format(accuracy_score(y_true, y_pred)))
    print('Test precision: {:.3f}'.format(precision_score(
        y_true, y_pred, pos_label=concept)))
    print('Test recall: {:.3f}'.format(
        recall_score(y_true, y_pred, pos_label=concept)))
    coef = np.squeeze(lr_model.coef_)
    nnotzero = np.sum(coef != 0)
    print('\nFeatures used: {:.3f}'.format(nnotzero))
    coef_sorted = np.argsort(-np.abs(coef))
    max_coef = coef_sorted[0]
    print('Largest features: {}'.format(coef_sorted[0:3]))

    # Plot relevant figures to results_dir.
    # Logistic regression weights.
    plt.figure()
    plt.plot([0, len(coef)-1], [0, 0], color='blue')
    plt.bar(np.arange(len(coef)), coef, width=5, color='blue')
    plt.xlabel('Weight dimension (neuron)')
    plt.ylabel('Weight value')
    plt.title('Logistic regression weights\nConcept {}'.format(concept))
    plt.savefig(os.path.join(
        results_dir, 'classifier_weights_{}_{}.png'.format(concept,
                                                           state_type)))

    # Concept unit
    # Use the training data here, because it has more data to show.
    # There is no implication in using it, since we are not comparing
    # the results with the logistic regression model.
    plt.figure()
    concept_unit = np.array(trX)[:, max_coef]
    Y_index_0 = [i for i, e in enumerate(trY) if e == concept]
    Y_index_1 = [i for i, e in enumerate(trY) if e != concept]
    concept_unit_0 = concept_unit[Y_index_0]
    # weigths_ guarantees that the maximum y is 1.0
    weights_0 = np.ones_like(concept_unit_0)/float(len(concept_unit_0))
    plt.hist(concept_unit_0, weights=weights_0, facecolor='blue',
             alpha=0.3, label='match')
    concept_unit_1 = concept_unit[Y_index_1]
    weights_1 = np.ones_like(concept_unit_1)/float(len(concept_unit_1))
    plt.hist(concept_unit_1, weights=weights_1, facecolor='red',
             alpha=0.3, label='no-match')
    plt.xlabel('Activation')
    plt.title('Distribution of the activation of Neuron \
              \nConcept {}'.format(max_coef, concept))
    plt.legend()
    plt.savefig(os.path.join(
        results_dir, 'concept_unit_{}_{}.png'.format(concept, state_type)))

    # Close all figures to save memory
    plt.close('all')

    return lr_model


def compute_lrc_top_k(concept, lrc_model, X_t, X_t_pos_tags_, results_dir,
                      k=3):
    """
    Trains and saves logistic regression classifiers for the
    k best neurons = dimensions with k larges absolute weights.
    Outputs the metrics of the trained logistic regression classifiers.

    Args:
        concept (str): POS tag corresponding to the logistic regression
                       classifiers.
        X_t (list): list of ndarray with LSTM states.
        X_t_pos_tags_ (list): list of POS tags for each byte in X_t, with
                              '[concept]' and '[concept-NOT]' labels.
        results_dir (str): the path to save results to.
        k (int): number of top dimensions to use.

    """
    coef = np.squeeze(lrc_model.coef_)
    coef_sorted = np.argsort(-np.abs(coef))
    for i in np.arange(k):
        print('\n--- TOP {} dimensions'.format(str(i+1)))
        X_t_ = [x[coef_sorted[0:i+1]] for x in X_t]
        # Split data into train, valid, test sets
        trX, vaX, teX, trY, vaY, teY = split_train_valid_test(
            X_t_, X_t_pos_tags_)

        # Train a Logistic Regression Classifier on top of all dimensions.
        print('\nTraining a Logistic Regression Classifier...')
        lr_model = train_lrc(trX, trY, vaX, vaY, teX, teY)
        print('...completed.')

        lr_file = results_dir + '/log_reg_model_' + concept \
            + '_top' + str(i+1) + '.sav'
        pickle.dump(lr_model, open(lr_file, 'wb'))

        print('\n--> Logistic Regression results.')
        print('Train accuracy: {:.3f}'.format(lr_model.score(trX, trY)))
        print('Valid accuracy: {:.3f}'.format(lr_model.score(vaX, vaY)))
        print('Test  accuracy: {:.3f}'.format(lr_model.score(teX, teY)))
        print('Regularization coef: {:.3f}\n'.format(lr_model.C))

        y_true, y_pred = teY, lr_model.predict(teX)
        print('Test accuracy:  {:.3f}'.format(accuracy_score(y_true,
                                                             y_pred)))
        print('Test precision: {:.3f}'.format(
            precision_score(y_true, y_pred, pos_label=concept)))
        print('Test recall: {:.3f}'.format(
            recall_score(y_true, y_pred, pos_label=concept)))
        print('\nFeatures used: {:.3f}'.format(
            np.sum(lr_model.coef_ != 0)))
        print('Largest feature: {}'.format(
            np.argsort(-np.abs(lr_model.coef_))[0]))


def concept_neurons(args):
    """
    Trains logistic regression classifiers for different POS tags.
    For each logistic regression classifier plots its weights per dimension,
    and an histogram with the cell state value of the LSTM dimension
    corresponding to the largest weight of the classifier.
    These results are written to the folder specified in args.results_dir.

    Args:
        args (argparse): CLI arguments.

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

    # Build a logistic regression classifier for each concept
    # as a One-vs-All model.
    for concept in concepts:
        print('\n\n-----> CONCEPT: {}'.format(concept))
        sys.stdout.flush()

        # The lists are copied, because we need to make modifications. This way
        # the initial X_t and X_t_pos_tags are preserved.
        X_t_ = X_t
        X_t_pos_tags_ = X_t_pos_tags

        # Check if the concept exists in the dataset.
        if concept in X_t_pos_tags_:
            pass
        else:
            print('concept not found')
            continue

        # Replace tags different from concept with 'concept-NOT'.
        X_t_pos_tags_ = [concept if tag == concept else concept+'-NOT'
                         for tag in X_t_pos_tags_]

        # Split data into train, valid, test sets.
        trX, vaX, teX, trY, vaY, teY = split_train_valid_test(X_t_,
                                                              X_t_pos_tags_)

        lrc_model = compute_lrc(concept, X_t_, X_t_pos_tags_,
                                trX, vaX, teX, trY, vaY, teY,
                                results_dir)

        # Build classifiers for the k 'best' neurons.
        compute_lrc_top_k(concept, lrc_model, X_t, X_t_pos_tags_, results_dir,
                          k=3)


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
                        help='directory to save results')
    parser.add_argument('--group_tags', action='store_true',
                        help="""group all VB* tags into VB;
                                JJ* into JJ;
                                NN* into NN;
                                NNP* into NNP;
                                RB* into RB.
                        """)

    args = parser.parse_args()
    concept_neurons(args)
