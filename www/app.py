"""
2018, University of Freiburg.
Joao Carvalho <carvalhj@cs.uni-freiburg.de>
"""
import sys
import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from byte_LSTM.model import Model
from nltk import word_tokenize, pos_tag
from glob import glob
import re


# Docker maps /nfs/students/<firstname>-<lastname> to /extern/data
extern_data = '/extern/data/'


"""
Grouped concepts
"""
concepts_grouped = ['(', ')', ',', '.', 'CC', 'CD', 'DT', 'IN', 'JJ', 'MD',
                    'NN', 'NNP', 'PRP', 'RB', 'TO', 'VB']

"""
Restore pre-trained tensorflow model for Wikitext
"""
print('Loading models...')
trained_model = os.path.join(
    extern_data, 'byte_LSTM_trained_models/wikitext/save/95/')
model_path = os.path.abspath(trained_model)
with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
    model_saved_args = pickle.load(f)

# Create an instance of the RNN tensorflow model.
tf.reset_default_graph()
lstm_model = Model(model_saved_args, sampling=True)

tf_sess = tf.Session()
tf_sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
if not tf.train.get_checkpoint_state(model_path):
    print('Unable to find checkpoint file.')
    sys.exit()
model_path = os.path.join(
    model_path, tf.train.get_checkpoint_state(model_path)
                    .model_checkpoint_path.split('/')[-1])
saver.restore(tf_sess, model_path)
print('...completed.')

# Load pre-trained logistic regression classifiers
dataset_options = ['group_tags_250_lines', 'not_group_tags_250_lines',
                   'group_tags_500_lines', 'not_group_tags_500_lines',
                   'group_tags_nltk_data_1000',
                   'not_group_tags_nltk_data_1000']

lr_classifiers = {}
for dataset in dataset_options:
    lr_classifiers[dataset] = {}
    log_reg_dir = 'static/results/' + dataset + '/'
    log_reg_files = glob(os.path.join(log_reg_dir, 'log_reg_model_*'))
    for log_reg_file in log_reg_files:
        lr_classifiers[dataset][log_reg_file] = pickle.load(open(log_reg_file,
                                                                 'rb'))

"""
Restore pre-trained tensorflow model for Amazon product reviews
"""
print('Loading models...')
trained_model = os.path.join(
    extern_data, 'byte_LSTM_trained_models/amazon/save/806/')
model_path = os.path.abspath(trained_model)
with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
    model_saved_args = pickle.load(f)

# Create an instance of the RNN tensorflow model.
tf.reset_default_graph()
lstm_model_amazon = Model(model_saved_args, sampling=True)

tf_sess_amazon = tf.Session()
tf_sess_amazon.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
if not tf.train.get_checkpoint_state(model_path):
    print('Unable to find checkpoint file.')
    sys.exit()
model_path = os.path.join(
    model_path, tf.train.get_checkpoint_state(model_path)
                    .model_checkpoint_path.split('/')[-1])
saver.restore(tf_sess_amazon, model_path)
print('...completed.')

# Load Logistic Regression Classifier
lr_model_amazon = pickle.load(open('models/amazon/log_reg_model.sav', 'rb'))

"""
Flask web app
"""
app = Flask(__name__, static_url_path='')


# Base index page
@app.route('/')
def index_page():
    return app.send_static_file('index.html')


# Sample the pre-trained rnn model
@app.route('/sample', methods=['POST'])
def sample_model():
    # Parse request
    n_samples = int(request.form['n_samples'])
    prime = str(request.form['prime'])
    if not prime:
        prime = '\n'
    sample_type = int(request.form['sample_type'])
    temperature = float(request.form['temperature'])

    # Sample the model
    pred_bytes, _, _, _ = lstm_model.sample(
        tf_sess, n_samples=n_samples, prime=prime, sample_type=sample_type,
        temperature=temperature)

    response_msg = pred_bytes.decode(encoding='utf-8', errors='ignore')

    return jsonify(response_msg)


# Set dataset for logistic regression classifier
@app.route('/set_dataset', methods=['POST'])
def set_dataset():
    # Send image paths of results
    dataset = str(request.form['dataset'])
    table_results_html = open('static/results/' + dataset +
                              '/table_results.html', 'r').read()

    response = {}
    response["table_results_html"] = table_results_html

    return jsonify(response)


# Retrieve concept neuron images
@app.route('/concept_plots', methods=['POST'])
def concept_neuron_results():
    # Send image paths of results
    dataset = str(request.form['dataset'])
    concept = str(request.form['concept'])
    lr_weights_img_path = '/static/results/' + dataset + \
                          '/classifier_weights_' + concept + '_cell_states.png'
    concept_neuron_hist_img_path = '/static/results/' + dataset + \
                                   '/concept_unit_' + concept + \
                                   '_cell_states.png'

    response = {}
    response["lr_weights"] = lr_weights_img_path
    response["concept_neuron_hist"] = concept_neuron_hist_img_path

    return jsonify(response)


# Return the cell states of the input text
@app.route('/sample_concept_neuron', methods=['POST'])
def sample_concept_neuron():
    input_text = str(request.form['input_text'])
    neuron = int(request.form['neuron'])
    dataset = str(request.form['dataset'])
    if 'not_group' in dataset:
        group_concepts = False
    else:
        group_concepts = True

    # Split text into parts, separated by newline
    input_text_split = input_text.split('\n')
    input_text_pos_tag = []
    for sentence in input_text_split:
        if sentence != '':  # For cases like \n\n
            tokens = word_tokenize(sentence, language='english')
            input_text_pos_tag.append(pos_tag(tokens, lang='eng'))

    # Preprocess before transforming the text in the RNN
    response = {}
    response['input_text'] = []
    response['cell_states'] = []
    response['pos_tag'] = []
    for sentence_pos_tag in input_text_pos_tag:
        s = ''
        pos_tag_sentence = []
        for word, tag in sentence_pos_tag:
            s += word + ' '
            for _ in range(len(word.encode('utf-8'))):
                # Group tags if the option is selected
                if group_concepts:
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

                    if tag in concepts_grouped:
                        pass
                    else:
                        tag = 'OTHER'

                pos_tag_sentence.append(tag)

            pos_tag_sentence.append('SPACE')

        input_text_t = lstm_model.transform(tf_sess, [s])

        input_t_str = input_text_t['X_bytes'][0].decode('utf-8')
        input_t_states = [state.tolist()[neuron]
                          for sublist in input_text_t['cell_states']
                          for state in sublist]
        pos_tags = []

        # If a unicode char has more than one byte, take the state associated
        # with the last byte.
        i = 0
        input_t_states_tmp = []
        for ch in input_t_str:
            n_bytes = len(ch.encode('utf-8'))
            if n_bytes == 1:
                input_t_states_tmp.append(input_t_states[i])
                pos_tags.append(pos_tag_sentence[i])
                i += 1
            else:
                input_t_states_tmp.append(input_t_states[i + n_bytes - 1])
                pos_tags.append(pos_tag_sentence[i + n_bytes - 1])
                i += n_bytes

        response['input_text'].append(input_t_str)
        response['cell_states'].append(input_t_states_tmp)
        response['pos_tag'].append(pos_tags)

    return jsonify(response)


# Return the cell states of the input text
@app.route('/sample_concept_classifier', methods=['POST'])
def sample_concept_classifier():
    input_text = str(request.form['input_text'])
    dataset = str(request.form['dataset'])
    concept = str(request.form['concept'])
    concept_classifier = str(request.form['concept_classifier'])
    if 'not_group' in dataset:
        group_concepts = False
    else:
        group_concepts = True

    # Load the logistic regression classifier of all dimension to get the
    # largest weights
    lr_model_all_file = 'static/results/' + dataset + \
                        '/log_reg_model_' + concept + '_all.sav'
    lr_model_all = lr_classifiers[dataset][lr_model_all_file]
    top_weights = np.argsort(-np.abs(lr_model_all.coef_)).squeeze()

    # Load logistic regression classifier
    lr_model_file = 'static/results/' + dataset + '/log_reg_model_' + \
                    concept + '_' + concept_classifier + '.sav'
    lr_model = lr_classifiers[dataset][lr_model_file]
    if '-NOT' in lr_model.classes_[0]:
        concept_class_index = 1
    else:
        concept_class_index = 0

    # Split text into parts, separated by newline
    input_text_split = input_text.split('\n')
    input_text_pos_tag = []
    for sentence in input_text_split:
        if sentence != '':  # For cases like \n\n
            tokens = word_tokenize(sentence, language='english')
            input_text_pos_tag.append(pos_tag(tokens, lang='eng'))

    # Preprocess before transforming the text in the RNN
    response = {}
    response['input_text'] = []
    response['probabilities'] = []
    response['pos_tag'] = []
    for sentence_pos_tag in input_text_pos_tag:
        s = ''
        pos_tag_sentence = []
        for word, tag in sentence_pos_tag:
            s += word + ' '
            for _ in range(len(word.encode('utf-8'))):
                if group_concepts:
                    # Preprocess tags
                    if re.match('VB.*$', tag):  # Group all verbs
                        tag = 'VB'
                    elif re.match('JJ.*$', tag):  # Group all adjectives
                        tag = 'JJ'
                    elif re.match('NN$|NNS$', tag):  # Group all nouns
                        tag = 'NN'
                    elif re.match('NNP$|NNPS$', tag):  # Group all proper
                        tag = 'NNP'
                    elif re.match('RB.*$', tag):  # Group all adverbs
                        tag = 'RB'

                    if tag in concepts_grouped:
                        pass
                    else:
                        tag = 'OTHER'

                pos_tag_sentence.append(tag)
            pos_tag_sentence.append('SPACE')

        input_text_t = lstm_model.transform(tf_sess, [s])

        input_t_str = input_text_t['X_bytes'][0].decode('utf-8')

        # Predict probability of each each byte belonging to the concept
        if 'all' in lr_model_file:
            states = [state for sublist in input_text_t['cell_states']
                      for state in sublist]
            probabilities = [lr_model.predict_proba(state.reshape(1, -1))
                             .squeeze()[concept_class_index]
                             for state in states]
        elif 'top1' in lr_model_file:
            states = [state[top_weights[0]]
                      for sublist in input_text_t['cell_states']
                      for state in sublist]
            probabilities = [lr_model.predict_proba(state.reshape(1, -1))
                             .squeeze()[concept_class_index]
                             for state in states]
        elif 'top2' in lr_model_file:
            states = [state[top_weights[0:2]]
                      for sublist in input_text_t['cell_states']
                      for state in sublist]
            probabilities = [lr_model.predict_proba(state.reshape(1, -1))
                             .squeeze()[concept_class_index]
                             for state in states]
        elif 'top3' in lr_model_file:
            states = [state[top_weights[0:3]]
                      for sublist in input_text_t['cell_states']
                      for state in sublist]
            probabilities = [lr_model.predict_proba(state.reshape(1, -1))
                             .squeeze()[concept_class_index]
                             for state in states]

        pos_tags = []

        # If a unicode char has more than one byte, take the state associated
        # with the last byte.
        i = 0
        probabilities_tmp = []
        for ch in input_t_str:
            n_bytes = len(ch.encode('utf-8'))
            if n_bytes == 1:
                probabilities_tmp.append(probabilities[i])
                pos_tags.append(pos_tag_sentence[i])
                i += 1
            else:
                probabilities_tmp.append(probabilities[i + n_bytes - 1])
                pos_tags.append(pos_tag_sentence[i + n_bytes - 1])
                i += n_bytes

        response['input_text'].append(input_t_str)
        response['probabilities'].append(probabilities_tmp)
        response['pos_tag'].append(pos_tags)

    return jsonify(response)


# Sample a review
@app.route('/sample_reviews', methods=['POST'])
def sample_reviews(sent_neuron=981):
    n_samples = int(request.form['n_samples'])
    prime = '\n'
    sample_type = int(request.form['sample_type'])
    temperature = float(request.form['temperature'])

    pred_bytes, pred_probs, _, cell_states = lstm_model_amazon.sample(
        tf_sess_amazon, n_samples=n_samples, prime=prime,
        sample_type=sample_type, temperature=temperature)

    response = list()
    for byte, probs, state in zip(pred_bytes, pred_probs, cell_states):
        if isinstance(probs, int):
            probs = np.array([1])
        response.append([[chr(byte)], probs.tolist(),
                         state.tolist()[sent_neuron]])

    return jsonify(response)


# Classify a review with the sentiment neuron
@app.route('/classify_review', methods=['POST'])
def classify_review(sent_neuron=981):
    review_text = str(request.form['review_text'])

    # Preprocess.
    review_text = review_text.replace('\n', ' ')

    # Get the cell state after processing each byte.
    tr_review = lstm_model_amazon.transform(tf_sess_amazon, [review_text])

    tr_review_neuron = [state.tolist()[sent_neuron]
                        for sublist in tr_review['cell_states']
                        for state in sublist]

    # Predict with the Logistic Regression Classifer.
    tr_review_lr = tr_review['cell_states'][0][np.newaxis, :]
    pred_probability = lr_model_amazon.predict_proba(tr_review_lr[-1])

    data = {"tr_review_neuron": tr_review_neuron,
            "pred_probability": pred_probability.tolist()[-1]}

    return jsonify(data)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
