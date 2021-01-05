import unittest
from ID3 import ID3ContinuousFeatures


class TestSolver(unittest.TestCase):

    def test_binary_one_divide(self):
        classifier = ID3ContinuousFeatures.get_classify("./test_csv/binary_one_divide.csv")
        self.assertTrue(classifier == (2, [(0, [], 1), (0, [], 0)], 1))

    def test_small_binary(self):
        classifier = ID3ContinuousFeatures.get_classify("./test_csv/small_binary.csv")
        self.assertTrue(classifier == ((1, [(3, [(2, [(0, [], 1), (0, [], 0)], 1), (0, [], 1)], 1), (3, [(0, [], 1), (0, [], 0)], 1)], 1)))

if __name__ == '__main__':
    unittest.main()
