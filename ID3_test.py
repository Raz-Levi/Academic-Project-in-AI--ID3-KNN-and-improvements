import unittest
from ID3 import *
from utils import create_test
import os


class TestID3(unittest.TestCase):

    def test_binary_one_divide(self):
        test_path = "./test_csv/binary_one_divide.csv"
        classifier = ID3ContinuousFeatures.get_classifier(test_path)
        self.assertTrue(ID3ContinuousFeatures.get_accuracy(classifier, test_path))

    def test_small_binary(self):
        test_path = "./test_csv/small_binary.csv"
        classifier = ID3ContinuousFeatures.get_classifier(test_path)
        self.assertTrue(ID3ContinuousFeatures.get_accuracy(classifier, test_path))

    def test_binary_check_ig(self):
        test_path = "./test_csv/binary_check_ig.csv"
        classifier = ID3ContinuousFeatures.get_classifier(test_path)
        self.assertTrue(ID3ContinuousFeatures.get_accuracy(classifier, test_path))

    def test_binary_check_noise(self):
        test_path = "./test_csv/binary_check_noise.csv"
        classifier = ID3ContinuousFeatures.get_classifier(test_path)
        self.assertTrue(ID3ContinuousFeatures.get_accuracy(classifier, test_path))

    def test_binary_random_test(self):
        test_path = "./test_csv/random_test.csv"
        classifier = ID3ContinuousFeatures.get_classifier(test_path)
        self.assertTrue(ID3ContinuousFeatures.get_accuracy(classifier, test_path))

    def test_binary_all_noise(self):
        test_path = "./test_csv/binary_all_noise.csv"
        classifier = ID3ContinuousFeatures.get_classifier(test_path)
        self.assertTrue(ID3ContinuousFeatures.get_accuracy(classifier, test_path))

    def test_randomly(self):
        classifier = ID3ContinuousFeatures.get_classifier(create_test(1000, 1000))
        self.assertTrue(ID3ContinuousFeatures.get_accuracy(classifier, "./test_csv/try.csv"))
        os.remove("./test_csv/try.csv")

    def test_monster(self):
        for _ in range(3):
            classifier = ID3ContinuousFeatures.get_classifier(create_test(100, 100))
            self.assertTrue(ID3ContinuousFeatures.get_accuracy(classifier, "./test_csv/try.csv"))
        os.remove("./test_csv/try.csv")

    # def test_monster_accuracy(self):
    #     accuracy = []
    #     for _ in range(10):
    #         iteration = ID3ContinuousFeatures.learn(create_test(1000, 1000, "train"), create_test(100, 1000, "test"))
    #         accuracy.append(iteration)
    #         self.assertTrue(0 <= iteration <= 1)
    #     print(max(accuracy))
    #     os.remove("./test_csv/train.csv")
    #     os.remove("./test_csv/test.csv")

    # def test_big_accuracy(self):
    #     print(ID3ContinuousFeatures.learn(create_test(10000, 10000, "train"), create_test(100, 10000, "test")))
    #     os.remove("./test_csv/train.csv")
    #     os.remove("./test_csv/test.csv")


if __name__ == '__main__':
    unittest.main()
