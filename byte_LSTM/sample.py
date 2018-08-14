import argparse
import pickle
from model import Model
import os
import tensorflow as tf
import sys


class Sample:
    def __init__(self, save_dir):
        """
        Restore a pre-trained tensorflow session to sample from.

        Args:
            save_dir (str): directory with saved model.

        Attributes:
            model (Model): base model built in tensorflow.
            sess (tf.Session): tensorflow session with pre-trained model
                               variables.
        """
        # Load configuration arguments.
        save_dir = os.path.abspath(save_dir)
        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            model_saved_args = pickle.load(f)

        # Create an instance of the RNN tensorflow model.
        tf.reset_default_graph()
        self.model = Model(model_saved_args, sampling=True)
        # Restore the saved session.
        sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if not ckpt:
            print('Unable to find checkpoint file.')
            sys.exit()
        model_path = os.path.join(
            save_dir, ckpt.model_checkpoint_path.split('/')[-1])
        saver.restore(sess, model_path)
        self.sess = sess

    def sample(self, n, prime, sample_type, temperature):
        """
        Samples a RNN model.

        Args:
            n (int): number of bytes to sample.
            prime (str): initial string to sample from.
            sample_type (int): 0 - sample with max probability;
                               1 - sample from a multinomial distribution.
            temperature (float): softmax temperature (0,1]

        Returns:
            predicted_bytes (list): list of predicted bytes.
            predicted_probs (list): list of predicted probabilities for each
                                    byte.
            hidden_states (list): list of produced hidden states for the whole
                                  sequence.
            cell_states (list): list of produced cell states for the whole
                                sequence.
        """
        predicted_bytes, predicted_probs, \
            hidden_states, cell_states = self.model.sample(
                self.sess, n, prime, sample_type, temperature)

        return predicted_bytes, predicted_probs, hidden_states, cell_states


def main():
    """
    Parse CLI arguments.
    Samples a previously trained LSTM model.
    """
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory with saved model')
    parser.add_argument('--n', type=int, default=1,
                        help='number of bytes to sample')
    parser.add_argument('--prime', type=str, default='\n',
                        help='text to sample after')
    parser.add_argument('--sample_type', type=int, default=1,
                        help='0 - take the max probability; 1 - sample from \
                              a multinomial distribution')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for softmax predictions (0,1]')

    args = parser.parse_args()

    sample = Sample(args.save_dir)
    predicted_bytes, _, _, _ = sample.sample(
        args.n, args.prime, args.sample_type, args.temperature)

    sys.stdout.write(predicted_bytes.decode('utf-8', errors='replace'))
    print()


if __name__ == '__main__':
    main()
