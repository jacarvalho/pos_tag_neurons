import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm


class Model():

    def __init__(self, args, sampling=False):
        """
        Builds a tensorflow model to train a one-layer byte-LSTM.
        This model is a representation of the computational graph of a LSTM.
        It can be used for training and sampling, by changing the batch size
        and sequence length.

        Args:
            See arguments of train.py.
            sampling (bool): if sampling is True, the model is used for
                             sampling, otherwise for training.

        Atrributes:
            args (argparse): training/sampling arguments.
            batch_size (int): mini-batch size.
            seq_length (int): steps used for truncated backpropagation.
            rnn_size (int): size of LSTM hidden and cell states.
            global_step (tf.int32): global training step.
            batchX_placeholder (tf.placeholder): placeholder for input data.
            batchY_placeholder (tf.placeholder): placeholder for target data.
            initial_hidden_state (tf.placeholder): LSTM initial hidden state.
            initial_cell_state (tf.placeholder): LSTM initial cell state.
            final_hidden_state (tf.placeholder): LSTM final hidden state
                after processing the mini-batch.
            final_cell_state (tf.placeholder): LSTM final cell state
                after processing the mini-batch.
            logits_series (list of tf.tensor): list of logits.
            predict_series (list of tf.tensor): list of predictions (softmax
                of logits).
            total_loss (float): average cross-entropy loss of the mini-batch.
            loss_summary (tf.summary.scalar): tf.summary to store the loss.
            accuracy (float): average accuracy of the mini-batch.
            accuracy_summary (tf.summary.scalar): tf.summary to store the
                accuracy.
            lr (float): non-trainable tf.Variable learning rate.
            train_step (tf.optimizer): tensorflow operation to apply gradients
                to trainable variables.
        """
        self.args = args
        # When the model is being sampled, process one byte at a time.
        if sampling:
            self.batch_size = 1
            self.seq_length = 1
        else:
            self.batch_size = args.batch_size
            self.seq_length = args.seq_length
        self.rnn_size = args.rnn_size

        # global_step variable that is incremented every time a
        # mini-batch is processed.
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                       name='global_step')

        with tf.name_scope('input'):
            self.batchX_placeholder = tf.placeholder(
                tf.int32, [self.batch_size, self.seq_length], name='input')
            # Splits the sequence of bytes into one byte sequences, since they
            # are processed one at a time, when computing the LSTM states.
            # input_series:
            #     len(input_series) = seq_length
            #     input_series[i].shape = [batch_size, 1]
            inputs_series = tf.unstack(self.batchX_placeholder, axis=1)
            tf.summary.histogram('input_bytes', self.batchX_placeholder)

        with tf.name_scope('target'):
            self.batchY_placeholder = tf.placeholder(
                tf.int32, [self.batch_size, self.seq_length], name='target')
            targets_series = tf.unstack(self.batchY_placeholder, axis=1)

        # Using a variable scope, ensures the model reuses the variables
        # (weights and biases) between different batches.
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            # Build the LSTM model and perform the forward passes.
            self.initial_hidden_state = tf.placeholder(
                tf.float32, [self.batch_size, self.rnn_size],
                name='initial_hidden_state')

            self.initial_cell_state = tf.placeholder(
                tf.float32, [self.batch_size, self.rnn_size],
                name='initial_cell_state')

            # Forget gate.
            W_f = tf.get_variable(
                name='W_f', shape=[256 + self.rnn_size, self.rnn_size],
                dtype=tf.float32)
            tf.summary.histogram('W_f', W_f)

            b_f = tf.get_variable(name='b_f', shape=[self.rnn_size],
                                  dtype=tf.float32)
            tf.summary.histogram('b_f', b_f)

            # Input gate.
            W_i = tf.get_variable(
                name='W_i', shape=[256 + self.rnn_size, self.rnn_size],
                dtype=tf.float32)
            tf.summary.histogram('W_i', W_i)

            b_i = tf.get_variable(name='b_i', shape=[self.rnn_size],
                                  dtype=tf.float32)
            tf.summary.histogram('b_i', b_i)

            # Tanh gate.
            W_c = tf.get_variable(
                name='W_c', shape=[256 + self.rnn_size, self.rnn_size],
                dtype=tf.float32)
            tf.summary.histogram('W_c', W_c)

            b_c = tf.get_variable(name='b_c', shape=[self.rnn_size],
                                  dtype=tf.float32)
            tf.summary.histogram('b_c', b_c)

            # Output gate.
            W_o = tf.get_variable(
                name='W_o', shape=[256 + self.rnn_size, self.rnn_size],
                dtype=tf.float32)
            tf.summary.histogram('W_o', W_o)

            b_o = tf.get_variable(name='b_o', shape=[self.rnn_size],
                                  dtype=tf.float32)
            tf.summary.histogram('b_o', b_o)

            # Forward pass.
            current_hidden_state = self.initial_hidden_state
            current_cell_state = self.initial_cell_state
            hidden_states_series = []
            cell_states_series = []
            for current_input in inputs_series:
                # Transform inputs to one-hot-encoded vector representing
                # a byte (256 positions).
                x_t = tf.one_hot(current_input, depth=256)

                # Concatenate the input and hidden sates.
                input_and_state_concat = tf.concat(
                    [x_t, current_hidden_state], axis=1)

                f_t = tf.sigmoid(tf.matmul(
                    input_and_state_concat, W_f) + b_f, name='xh_W_f')

                i_t = tf.sigmoid(tf.matmul(
                    input_and_state_concat, W_i) + b_i, name='xh_W_i')

                c_t_ = tf.tanh(tf.matmul(
                    input_and_state_concat, W_c) + b_c, name='xh_W_c')

                c_t = f_t * current_cell_state + i_t * c_t_
                current_cell_state = c_t
                cell_states_series.append(c_t)

                o_t = tf.sigmoid(tf.matmul(
                    input_and_state_concat, W_o) + b_o, name='xh_W_o')

                h_t = o_t * tf.tanh(c_t)
                current_hidden_state = h_t
                hidden_states_series.append(h_t)

            # Last states after processing the whole sequence.
            self.final_hidden_state = current_hidden_state
            tf.summary.histogram('final_hidden_state', self.final_hidden_state)
            self.final_cell_state = current_cell_state
            tf.summary.histogram('final_cell_state', self.final_cell_state)

        with tf.variable_scope('logits_softmax', reuse=tf.AUTO_REUSE):
            # Compute the output vector from the hidden state.
            W_out = tf.get_variable(
                name='W_out', shape=[self.rnn_size, 256], dtype=tf.float32)
            tf.summary.histogram('W_out', W_out)
            b_out = tf.get_variable(
                name='b_out', shape=[256], dtype=tf.float32)
            tf.summary.histogram('b_out', b_out)

            self.logits_series = [tf.matmul(h, W_out, name='h_W_out') + b_out
                                  for h in hidden_states_series]
            tf.summary.histogram('logits',
                                 tf.convert_to_tensor(self.logits_series))

            self.predict_series = [tf.nn.softmax(logits)
                                   for logits in self.logits_series]
            tf.summary.histogram('softmax_predictions',
                                 tf.convert_to_tensor(self.predict_series))

        with tf.name_scope('loss'):
            # Compute the cross entropy of each mini-batch.
            # Do not implement cross entropy (among others, it can lead to
            # numerical instability when computing a logarithm of 0).
            cross_entropies = [tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=batch_targets, logits=batch_logits)
                for batch_targets, batch_logits in
                zip(targets_series, self.logits_series)]
            tf.summary.histogram('cross_entropies',
                                 tf.convert_to_tensor(cross_entropies))

            # Average mini-batch loss.
            self.total_loss = tf.reduce_mean(cross_entropies)
            self.loss_summary = tf.summary.scalar('total_loss',
                                                  self.total_loss)

            # Compute accuracy.
            # Assume the correct byte is the one with higher probability after
            # applying softmax.
            predictions = [tf.argmax(prediction, axis=1)
                           for prediction in self.predict_series]
            correct_predictions = [
                tf.equal(tf.cast(prediction, tf.int32), target)
                for prediction, target in zip(predictions, targets_series)]
            # Average accuracy.
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, tf.float32))
            self.accuracy_summary = tf.summary.scalar(
                'accuracy', self.accuracy)

        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            # Tensorflow operation to compute and apply gradients.
            # lr is set to zero, but is assigned during training.
            self.lr = tf.Variable(0.0, trainable=False)
            # Clip the gradients, to prevent them from exploding when
            # backpropagating.
            tvars = tf.trainable_variables()
            grads_unclipped = tf.gradients(self.total_loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads_unclipped, args.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_step = optimizer.apply_gradients(
                zip(grads, tvars), global_step=self.global_step)

    def sample(self, sess, n_samples, prime, sample_type, temperature):
        """
        Samples a pre-trained LSTM tensorflow model.

        Args:
            sess (tf.Session): tensorflow session with a previously trained
                               LSTM model.
            n_samples (int): number of bytes to sample.
            prime (str): text (utf-8 encoded) to start sampling from.
            sample_type (int): 0 - take the max probability
                               1 - sample the next byte from a multinomial
                                   distribution.
            temperature (float): temperature for softmax predictions (0,1].

        Returns:
            predicted_bytes: a sequence of predicted bytes in a bytearray.
            predicted_probabilities: a sequence of probalities for the
                                     prediction of a byte.
            hidden_states (list): hidden states after processing each
                                       byte.
            cell_states (list): cell states after processing each byte.
        """
        # Compute the LSTM states until the penultimate byte.
        initial_hidden_state = np.zeros([1, self.rnn_size])
        initial_cell_state = np.zeros([1, self.rnn_size])
        [hidden_state, cell_state] = sess.run(
            [self.initial_hidden_state, self.initial_cell_state],
            feed_dict={self.initial_hidden_state: initial_hidden_state,
                       self.initial_cell_state: initial_cell_state})

        hidden_states = []
        cell_states = []
        prime_bytes = bytearray(prime.encode('utf-8', errors='ignore'))
        for byte in prime_bytes[:-1]:
            # For sampling, the input x is a placeholder of shape
            # (batch_size=1, seq_length=1)
            x = np.zeros((1, 1))
            x[0, 0] = byte
            feed = {self.batchX_placeholder: x,
                    self.initial_hidden_state: initial_hidden_state,
                    self.initial_cell_state: initial_cell_state}
            [hidden_state, cell_state] = sess.run([self.final_hidden_state,
                                                   self.final_cell_state],
                                                  feed)
            hidden_states.append(np.squeeze(hidden_state))
            cell_states.append(np.squeeze(cell_state))

        # Predict the next bytes.
        predicted_bytes = prime_bytes
        predicted_probs = [1.0 for _ in range(len(predicted_bytes) - 1)]
        input_byte = prime_bytes[-1]
        for _ in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = input_byte
            feed = {self.batchX_placeholder: x,
                    self.initial_hidden_state: hidden_state,
                    self.initial_cell_state: cell_state}
            [hidden_state, cell_state, logits] = sess.run(
                [self.final_hidden_state, self.final_cell_state,
                 self.logits_series],
                feed)

            hidden_states.append(np.squeeze(hidden_state))
            cell_states.append(np.squeeze(cell_state))

            # Apply softmax with temperature to logits.
            if temperature <= 0:
                probs = logits
            elif temperature > 0:
                scale = [logit / temperature for logit in logits]
                exp = np.exp(scale - np.max(scale))
                probs = exp / np.sum(exp)

            # Predicted byte.
            probs = np.squeeze(probs)
            predicted_probs.append(probs)
            if sample_type == 0:
                # Take the byte with maximum probability.
                predicted_byte = np.argmax(probs)
            if sample_type == 1:
                # Sample a byte from the multinomial distribution.
                predicted_byte = np.random.choice(range(len(probs)), p=probs)

            predicted_bytes.append(predicted_byte)
            input_byte = predicted_byte

        return predicted_bytes, predicted_probs, hidden_states, cell_states

    def transform(self, sess, X):
        """
        Transforms X into the hidden and cell states of an LSTM.

        Args:
            sess (tf.Session): tensorflow session with a previously trained
                               LSTM model.
            X (list): list of strings to transform.

        Returns:
            X_tr (dictionary of lists): utf-8 bytes in X, LSTM hidden and
                                        cell states.
        """
        tstart = time.time()
        X_bytes = []
        hidden_states = []
        cell_states = []
        for x_i in tqdm(X):
            hidden_state = np.zeros([1, self.rnn_size])
            cell_state = np.zeros([1, self.rnn_size])
            x_bytes = bytearray(x_i.encode('utf-8', errors='replace'))
            X_bytes.append(x_bytes)
            hidden_states_tmp = []
            cell_states_tmp = []
            for byte in tqdm(x_bytes):
                # For sampling, the input x is a placeholder of shape
                # (batch_size=1, seq_length=1)
                x = np.zeros((1, 1))
                x[0, 0] = byte
                feed = {self.batchX_placeholder: x,
                        self.initial_hidden_state: hidden_state,
                        self.initial_cell_state: cell_state}
                hidden_state, cell_state = sess.run([self.final_hidden_state,
                                                     self.final_cell_state],
                                                    feed)
                hidden_states_tmp.append(np.squeeze(hidden_state))
                cell_states_tmp.append(np.squeeze(cell_state))
            # Get all intermediate states
            hidden_states.append(np.squeeze(hidden_states_tmp))
            cell_states.append(np.squeeze(cell_states_tmp))

        print('{:.3f} seconds to transform {} examples'.format(
            time.time() - tstart, len(X)))

        X_tr = {'X_bytes': X_bytes, 'hidden_states': hidden_states,
                'cell_states': cell_states}
        return X_tr
