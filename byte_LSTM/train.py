import argparse
from model import Model
import tensorflow as tf
import numpy as np
import time
import os
import sys
from utils import TextLoader
import pickle


def main():
    """
    Parse CLI arguments.
    Trains the RNN model.
    """
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--option', type=str, default=None, required=True,
                        help='train, validate')
    parser.add_argument('--data_dir', type=str, default='data/tests',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from a model saved at this
                                path.
                                Must contain the files:
                                - chekpoint
                                - model.ckpt-*
                                - configs.pkl
                        """)
    parser.add_argument('--shard', type=int, default=0,
                        help='current shard of data')
    parser.add_argument('--rnn_size', type=int, default=4,
                        help='size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=3,
                        help='steps used for truncated backpropagation')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--lr_init', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', action='store_true',
                        help='linear learning rate decay')
    parser.add_argument('--train_bytes', type=int, default=None,
                        help="""number of training bytes, if lr_decay
                                is selected
                        """)
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='gradient clipping threshold')
    parser.add_argument('--print_every', type=int, default=100,
                        help='print training information every x steps')

    args = parser.parse_args()

    if args.lr_decay and args.train_bytes is None:
        parser.error('Provide total training bytes if lr_decay is selected.')

    train(args)


def train(args):
    """
    Trains a RNN model.

    Args:
        args (argparse): arguments to train the RNN.

    Returns:
        None.
    """
    s_time = time.time()

    # Check compatibility to continue training from a previous model.
    if args.init_from:
        assert os.path.isdir(args.init_from), \
            "{} does not exist".format(args.init_from)
        assert os.path.isfile(os.path.join(args.init_from, "config.pkl")), \
            "config.pkl file does not exist in path {}".format(args.init_from)

        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # Check if models are compatible.
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = pickle.load(f)
            check_list = ["rnn_size", "seq_length"]
            for check in check_list:
                assert vars(saved_model_args)[check] == vars(args)[check], \
                    "CLI argument and saved model disagree on %s".format(check)

    # Store configuration arguments.
    args.save_dir = os.path.join(args.save_dir, str(args.shard))
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Load input data.
    args.data_dir = os.path.join(args.data_dir, str(args.shard))
    if not os.path.isdir(args.data_dir):
        sys.exit('{} does not exist'.format(args.data_dir))
    args.data_dir = os.path.abspath(args.data_dir)
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)

    # Set logs directories.
    args.log_dir = os.path.join(args.log_dir, args.option)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    args.log_dir = os.path.abspath(args.log_dir)

    # Create an instance of the tensorflow model.
    tf.reset_default_graph()
    model = Model(args)

    with tf.Session() as sess:
        # Tensorboard summaries.
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        # Initialize variables (weigths and biases), with a Xavier uniform
        # initializer. See tf.glorot_uniform_initializer().
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()
        # Restore previous model and session.
        if args.init_from:
            saver.restore(sess, ckpt.model_checkpoint_path)

        losses, accuracies = [], []
        global_step_init = model.global_step.eval()
        # Run the model for training or validation/testing.
        for epoch_id in range(args.num_epochs):
            data_loader.reset_batch_pointer()

            # Reset the states at the beginning of each epoch.
            h_state = np.zeros([args.batch_size, args.rnn_size])
            c_state = np.zeros([args.batch_size, args.rnn_size])
            for batch_id in range(data_loader.num_batches):
                x, y = data_loader.next_batch()

                # Update the learning rate, with linear decay.
                global_step = model.global_step.eval()
                if args.lr_decay and global_step != 0:
                    total_weight_updates = args.train_bytes \
                        / (args.batch_size * args.seq_length)
                    lr = args.lr_init - (args.lr_init / total_weight_updates) \
                        * global_step
                    if lr < 1.5*10**-13:
                        lr = 1.5*10**-13
                else:
                    lr = args.lr_init
                sess.run(tf.assign(model.lr, lr))

                # Keep the states between batches to simulate full
                # backpropagation (stateful RNN).
                feed = {model.initial_hidden_state: h_state,
                        model.initial_cell_state: c_state,
                        model.batchX_placeholder: x,
                        model.batchY_placeholder: y}

                if args.option == 'train':
                    _, h_state, c_state, loss, accuracy, \
                        summary = sess.run([model.train_step,
                                            model.final_hidden_state,
                                            model.final_cell_state,
                                            model.total_loss,
                                            model.accuracy,
                                            summaries],
                                           feed_dict=feed)
                elif args.option == 'validate':
                    h_state, c_state, loss, accuracy, \
                        summary = sess.run([model.final_hidden_state,
                                            model.final_cell_state,
                                            model.total_loss,
                                            model.accuracy,
                                            summaries],
                                           feed_dict=feed)

                losses.append(loss)
                accuracies.append(accuracy)

                if args.option == 'train' and \
                        global_step % args.print_every == 0:
                    # Record training for tensorboard.
                    writer.add_summary(summary, global_step)
                    writer.flush()

                    print("Shard {} Epoch {}/{} Batch {}/{} ({}) -- "
                          "loss: {:.3f}, acc: {:.3f}"
                          .format(args.shard, epoch_id, args.num_epochs-1,
                                  batch_id, data_loader.num_batches-1,
                                  global_step, loss, accuracy))
                    sys.stdout.flush()

        # Save the model at the end of training.
        if args.option == 'train':
            save_model(args, sess, saver, global_step)

        # Save losses and accuracies.
        np.save(os.path.join(
            args.log_dir, 'loss_' + str(global_step_init)), losses)
        np.save(os.path.join(
            args.log_dir, 'accuracy_' + str(global_step_init)), accuracies)

        # Record training for tensorboard.
        writer.add_summary(summary, global_step)
        writer.flush()

        print("Shard {} Epoch {}/{} Batch {}/{} ({}) -- "
              "loss: {:.3f}, acc: {:.3f}"
              .format(args.shard, epoch_id, args.num_epochs-1,
                      batch_id, data_loader.num_batches-1,
                      global_step, loss, accuracy))
        sys.stdout.flush()

        # Record time spent.
        time_spent = time.time() - s_time
        hours, rem = divmod(time_spent, 3600)
        minutes, seconds = divmod(rem, 60)
        print('Train time: {:0>2}:{:0>2}:{:05.2f}'
              .format(int(hours), int(minutes), seconds))
        print('Time per batch: {:.3f}ms, time per byte: {:.3f}ms'.
              format(time_spent/(args.num_epochs *
                     data_loader.num_batches) * 1000,
                     time_spent/(args.num_epochs *
                     data_loader.num_batches *
                     args.batch_size * args.seq_length) * 1000))


def save_model(args, sess, saver, global_step):
    """
    Saves a tensorflow model.

    Args:
        args (arguments class): arguments to train the RNN.
        sess (tf.session): session to save and sample.
        saver (tf.saver): object to save tf trained variables.
        global_step (int): current global weight update step.

    Returns:
        None.
    """
    save_path = os.path.join(args.save_dir, 'model')
    saver.save(sess, save_path, global_step=global_step)
    print('Model saved to {}'.format(save_path))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
