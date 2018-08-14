"""
Copyright 2018 University of Freiburg
Joao Carvalho <carvalhj@cs.uni-freiburg.de>
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from glob import glob
import argparse


def plot_loss_accuracy(args):
    """
    Generates plots of loss and accuracies for training and validation data.

    Args:
        args (argparse class): contains the directory path where logs files
                               are stored.

    Returns:
        None.
    """
    logs_dir = os.path.abspath(args.log_dir)

    loss_files_train = glob(os.path.join(os.path.join(logs_dir, 'train'),
                                         'loss_*'))
    loss_files_valid = glob(os.path.join(os.path.join(logs_dir, 'validate'),
                                         'loss_*'))

    # Losses training
    indexes_train = []
    for f in loss_files_train:
        indexes_train.append(
            int(f.split('/')[-1].split('.')[0].split('_')[-1]))

    indexes_train.sort()
    losses_train = []
    accs_train = []
    for index in indexes_train:
        loss_train = np.load(os.path.join(
            logs_dir, 'train/loss_' + str(index) + '.npy'))
        mean_loss = np.mean(loss_train)
        losses_train.append(mean_loss)

        acc_train = np.load(os.path.join(
            logs_dir, 'train/accuracy_' + str(index) + '.npy'))
        mean_acc = np.mean(acc_train)
        accs_train.append(mean_acc)

    indexes_train = np.array(indexes_train)
    losses_train = np.array(losses_train)
    accs_train = np.array(accs_train)

    # Losses validation
    indexes_valid = []
    for f in loss_files_valid:
        indexes_valid.append(
            int(f.split('/')[-1].split('.')[0].split('_')[-1]))

    indexes_valid.sort()
    losses_valid = []
    accs_valid = []
    for index in indexes_valid:
        loss_valid = np.load(os.path.join(
            logs_dir, 'validate/loss_' + str(index) + '.npy'))
        mean_loss = np.mean(loss_valid)
        losses_valid.append(mean_loss)

        acc_valid = np.load(os.path.join(
            logs_dir, 'validate/accuracy_' + str(index) + '.npy'))
        mean_acc = np.mean(acc_valid)
        accs_valid.append(mean_acc)

    indexes_valid = np.array(indexes_valid)
    losses_valid = np.array(losses_valid)
    accs_valid = np.array(accs_valid)

    """
    Plots
    """
    # Loss
    plt.figure(1)
    # Convert from nats to bits
    plt.plot(indexes_train, math.log(math.e, 2) * losses_train, lw=0.9,
             label='train')
    plt.plot(indexes_valid, math.log(math.e, 2) * losses_valid, lw=0.9,
             label='valid')
    plt.title('Cross-entropy loss')
    plt.xlabel('Update')
    plt.ylabel('Bits')
    # plt.ylim(ymin=1.05, ymax=2.0)
    # plt.xlim(xmin=0)
    plt.legend()
    plt.savefig('losses.png')

    # Accuracy
    plt.figure(2)
    # Convert from nats to bits
    plt.plot(indexes_train, 100 * accs_train, lw=0.9, label='train')
    plt.plot(indexes_valid, 100 * accs_valid, lw=0.9, label='valid')
    plt.title('Accuracies')
    plt.xlabel('Update')
    plt.ylabel('%')
    # plt.ylim(ymin=1.05, ymax=2.0)
    # plt.xlim(xmin=0)
    plt.legend()
    plt.savefig('accuracies.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory containing loss and accuracy files')

    args = parser.parse_args()

    plot_loss_accuracy(args)
