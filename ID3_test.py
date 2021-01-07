import unittest
from ID3 import *
from utils import create_test
import os


class TestID3(unittest.TestCase):

    def test_binary_one_divide(self):
        test_path = "./test_csv/binary_one_divide.csv"
        self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path) == 1)

    def test_small_binary(self):
        test_path = "./test_csv/small_binary.csv"
        self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path) == 1)

    def test_binary_check_ig(self):
        test_path = "./test_csv/binary_check_ig.csv"
        self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path) == 1)

    def test_binary_check_noise(self):
        test_path = "./test_csv/binary_check_noise.csv"
        self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path))

    def test_binary_random_test(self):
        test_path = "./test_csv/random_test.csv"
        self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path) == 1)

    def test_binary_all_noise(self):
        test_path = "./test_csv/binary_all_noise.csv"
        self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path))

    def test_randomly(self):
        test_path = create_test(1000, 1000)
        self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path) == 1)
        os.remove(test_path)

    def test_monster(self):
        for _ in range(3):
            test_path = create_test(100, 100)
            self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path) == 1)
        os.remove("./test_csv/try.csv")

    def test_accuracy(self):
        print(ID3ContinuousFeatures.learn_without_pruning(create_test(1000, 1000, "train"), create_test(100, 1000, "test")))
        os.remove("./test_csv/train.csv")
        os.remove("./test_csv/test.csv")

    def test_pruning(self):
        create_test(100, 100, "train")
        create_test(10, 100, "test")
        print(ID3ContinuousFeatures.learn_without_pruning("./test_csv/train.csv", "./test_csv/test.csv"))
        print(ID3ContinuousFeatures.learn_with_pruning("./test_csv/train.csv", "./test_csv/test.csv"))
        os.remove("./test_csv/train.csv")
        os.remove("./test_csv/test.csv")

    # def test_monster_accuracy(self):
    #     accuracy = []
    #     for _ in range(10):
    #         iteration = ID3ContinuousFeatures.learn_without_pruning(create_test(1000, 1000, "train"), create_test(100, 1000, "test"))
    #         accuracy.append(iteration)
    #         self.assertTrue(0 <= iteration <= 1)
    #     print(max(accuracy))
    #     os.remove("./test_csv/train.csv")
    #     os.remove("./test_csv/test.csv")


if __name__ == '__main__':
    unittest.main()
