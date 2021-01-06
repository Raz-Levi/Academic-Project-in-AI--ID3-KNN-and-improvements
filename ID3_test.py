import unittest
from ID3 import ID3ContinuousFeatures
from random import randint


class TestSolver(unittest.TestCase):

    def test_binary_one_divide(self):
        classifier = ID3ContinuousFeatures.get_classify("./test_csv/binary_one_divide.csv")
        self.assertTrue(classifier == (2, [(0, [], 1), (0, [], 0)], 1))

    def test_small_binary(self):
        classifier = ID3ContinuousFeatures.get_classify("./test_csv/small_binary.csv")
        self.assertTrue(classifier == ((1, [(3, [(2, [(0, [], 1), (0, [], 0)], 1), (0, [], 1)], 1), (3, [(0, [], 1), (0, [], 0)], 1)], 1)))

    def test_binary_check_ig(self):
        csv_path = "./test_csv/binary_check_ig.csv"
        classifier = ID3ContinuousFeatures.get_classify(csv_path)
        self.assertTrue(classifier == (7, [(0, [], 0), (0, [], 1)], 1))

    def test_binary_check_noise(self):
        csv_path = "./test_csv/binary_check_noise.csv"
        classifier = ID3ContinuousFeatures.get_classify(csv_path)
        self.assertTrue(classifier == (7, [(0, [], 1), (0, [], 0)], 0))

    def test_binary_random_test(self):
        csv_path = "./test_csv/random_test.csv"
        classifier = ID3ContinuousFeatures.get_classify(csv_path)
        self.assertTrue(classifier == (4, [(0, [], 1), (5, [(1, [(0, [], 1), (0, [], 0)], 1), (0, [], 0)], 0)], 0))

    def test_binary_all_noise(self):
        csv_path = "./test_csv/binary_all_noise.csv"
        classifier = ID3ContinuousFeatures.get_classify(csv_path)
        self.assertTrue(classifier == (0, [], 1))


if __name__ == '__main__':
    unittest.main()
