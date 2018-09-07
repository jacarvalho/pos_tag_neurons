"""
Copyright 2018 University of Freiburg
Joao Carvalho <carvalhj@cs.uni-freiburg.de>
"""
import os
import numpy as np
import sys


class TextLoader():

    def __init__(self, data_dir, batch_size, seq_length):
        """
        Reads the content of a text file byte-wise into a tensor.
        Creates batches (X, y).

        Args:
            data_dir (str): directory where 'input.txt' is stored.
            batch_size (int): number of batches of size seq_length, used for
                              the computation of stochastic gradient descent.
            seq_length (int): lenght of the sequence processed by the RNN,
                              during the unrolled backpropagation.

        Atrributes:
            data_dir (str): directory where 'input.txt' is stored.
            batch_size (int): number of batches of size seq_length, used for
                              the computation of stochastic gradient descent.
            seq_length (int): lenght of the sequence processed by the RNN,
                              during the unrolled backpropagation.
            tensor (ndarray): stores the content of the input.txt file
                              byte-wise.
            num_batches (int): number of batches produced.
            x_batches (list of ndarray): batches of X.
                                         len(X) = num_batches
                                         X[i].shape = (batch_size, 1)
            y_batches (list of ndarray): batches of y.
                                         len(y) = num_batches
                                         y[i].shape = (batch_size, 1)
            batch_pointer (int): current (X, y) batch.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        tensor_file = os.path.join(data_dir, "input_bytes.npy")

        if not os.path.exists(tensor_file):
            self.preprocess(input_file, tensor_file)
        else:
            print("Loading preprocessed files...")
            sys.stdout.flush()
            self.load_preprocessed(tensor_file)
            print("...completed.")
            sys.stdout.flush()

        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, tensor_file):
        """
        Opens a text file and read its content byte-wise into a tensor file.

        Args:
            input_file (str): path of 'input.txt' file to read.
            tensor_file (str): path of tensor file to write to.

        Returns:
            None.
        """
        print("Reading text file...")
        sys.stdout.flush()
        try:
            with open(input_file, "rb") as f:
                self.tensor = np.fromfile(f, dtype='uint8')
        except OSError as e:
            print(e)

        print("...completed.")
        sys.stdout.flush()

        # The last byte of a file is a EOF. Remove it.
        self.tensor = self.tensor[:-1]

        print("Saving tensor to file...")
        sys.stdout.flush()
        np.save(tensor_file, self.tensor)
        print("...completed.")
        sys.stdout.flush()

    def load_preprocessed(self, tensor_file):
        """
        Opens a pre-loaded tensor file.

        Args:
            tensor_file (str): path of tensor file to open.

        Returns:
            None.
        """
        self.tensor = np.load(tensor_file)

    def create_batches(self):
        """
        Creates batches of training and vaidation data to feed the RNN.

        Args:
            None.

        Returns:
            None.
        """
        print("Creating batches...")
        sys.stdout.flush()

        x = self.tensor
        y = np.copy(self.tensor)
        y[:-1] = x[1:]
        y[-1] = x[0]

        self.num_batches = int(len(x) / (self.batch_size * self.seq_length))

        # Alert for a small dataset
        if self.num_batches == 0:
            assert False, print("Not enough training data. Make seq_length "
                                "and batch_size smaller.")

        # Select data to evenly divide arrays.
        x = x[:self.num_batches * self.batch_size * self.seq_length]
        y = y[:self.num_batches * self.batch_size * self.seq_length]

        self.x_batches = np.split(x.reshape(self.batch_size, -1),
                                  self.num_batches, axis=1)
        self.y_batches = np.split(y.reshape(self.batch_size, -1),
                                  self.num_batches, axis=1)

        print("...completed.")
        sys.stdout.flush()

    def next_batch(self):
        """
        Gets the next batch (x, y) in the sequence.

        Args:
            None.

        Returns:
            x (ndarray): Input data.
            y (ndarray): Target data.
        """
        x = self.x_batches[self.batch_pointer]
        y = self.y_batches[self.batch_pointer]
        self.batch_pointer += 1
        return x, y

    def reset_batch_pointer(self):
        """
        Resets pointers to training and validation batches.

        Args:
            None.

        Returns:
            None.
        """
        self.batch_pointer = 0
