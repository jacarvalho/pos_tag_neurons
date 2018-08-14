import unittest
import numpy as np
from utils import TextLoader

class utils_tests(unittest.TestCase):

    def test_create_batches_1(self):
        """
        Test creation of batches.
        """
        # Test contains 'abcdefhij', 10 chars
        data_loader = TextLoader(data_dir="tests/data/", batch_size=1,
                                 seq_length=3)
        self.assertEqual(data_loader.num_batches, 3)
        x_batches = [np.array([[97, 98,  99]]), 
                     np.array([[100, 101, 102]]),
                     np.array([[103, 104, 105]])]
        y_batches = [np.array([[98, 99,  100]]), 
                     np.array([[101, 102, 103]]),
                     np.array([[104, 105, 106]])]
                         
        self.assertEqual(len(data_loader.x_batches), len(x_batches))
        for array_test, array in zip(x_batches,
                                     data_loader.x_batches):
            self.assertEqual(True, (array_test == array).all())

        self.assertEqual(len(data_loader.y_batches), len(y_batches))
        for array_test, array in zip(y_batches,
                                     data_loader.y_batches):
            self.assertEqual(True, (array_test == array).all())

    def test_create_batches_2(self):
        """
        Test creation of batches.
        """
        # Test contains 'abcdefhij', 10 chars
        data_loader = TextLoader(data_dir="tests/data/", batch_size=3,
                                 seq_length=3)
        self.assertEqual(data_loader.num_batches, 1)
        x_batches = [np.array([[97, 98, 99],
                               [100, 101, 102],
                               [103, 104, 105]])]
        y_batches = [np.array([[98, 99, 100],
                               [101, 102, 103],
                               [104, 105, 106]])]
                         
        self.assertEqual(len(data_loader.x_batches), len(x_batches))
        for array_test, array in zip(x_batches,
                                     data_loader.x_batches):
            self.assertEqual(True, (array_test == array).all())

        self.assertEqual(len(data_loader.y_batches), len(y_batches))
        for array_test, array in zip(y_batches,
                                     data_loader.y_batches):
            self.assertEqual(True, (array_test == array).all())


if __name__ == '__main__':
    unittest.main()
