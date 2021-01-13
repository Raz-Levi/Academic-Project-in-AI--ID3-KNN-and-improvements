import unittest
from ID3 import *
from KNN import *
from KNNForest import *
from CostSensitiveKNN import *
from utils import *
import os


class TestID3(unittest.TestCase):
    def test_continuous(self):
        test_path = "./test_csv/continuous.csv"
        self.assertTrue(ID3ContinuousFeatures.classify_without_pruning(test_path, test_path) == 1)

    def test_medium_continuous(self):
        test_path = "./test_csv/medium_continuous.csv"
        self.assertTrue(ID3ContinuousFeatures.classify_without_pruning(test_path, test_path) == 1)

    def test_random(self):
        for _ in range(10):
            test_path = create_num_test(100, 100)
            self.assertTrue(ID3ContinuousFeatures.classify_without_pruning(test_path, test_path) > 0.95)
        os.remove("./test_csv/try.csv")

    def test_actual(self):
        self.assertTrue(ID3ContinuousFeatures.classify_without_pruning(TRAIN_PATH, TEST_PATH) == 0.9469026548672567)


class TestKNN(unittest.TestCase):
    def test_medium_continuous(self):
        test_path = "./test_csv/medium_continuous.csv"
        self.assertTrue(KNN(test_path).classify(test_path) == 1)
        self.assertFalse(KNN(test_path).classify_and_get_loss(test_path))

    def test_continuous(self):
        test_path = "./test_csv/continuous.csv"
        self.assertTrue(KNN(test_path).classify(test_path) == 1)
        self.assertFalse(KNN(test_path).classify_and_get_loss(test_path))

    def test_random(self):
        for _ in range(10):
            test_path = create_num_test(100, 100)
            self.assertTrue(KNN(test_path).classify(test_path) == 1)
            self.assertFalse(KNN(test_path).classify_and_get_loss(test_path))
        os.remove("./test_csv/try.csv")

    def test_actual(self):
        self.assertTrue(KNN(TRAIN_PATH).classify(TEST_PATH) == 0.9646017699115044)
        self.assertFalse(KNN(TRAIN_PATH).classify_and_get_loss(TRAIN_PATH))
        self.assertFalse(KNN(TEST_PATH).classify_and_get_loss(TEST_PATH))


class TestCostSensitiveKNN(unittest.TestCase):
    def test_actual(self):
        self.assertTrue(CostSensitiveKNN(TRAIN_PATH).classify(TEST_PATH) == 0.003539823008849558)

class TestKNNForest(unittest.TestCase):
    def test_countinius(self):
        train_path = "./test_csv/continuous.csv"
        i = 0
        for _ in range(1000):
            a = KNNForest(train_path, 1, 1, 1).classify(train_path)
            if a != 1:
                i += 1
        self.assertFalse(i)

    def test_actual(self):
        self.assertTrue(KNNForest(TRAIN_PATH, 1, 1, 1).classify(TEST_PATH) == 0.9469026548672567)

    def test_random_one(self):
        path = create_num_test(100,100)
        self.assertTrue(KNNForest(path, 1, 1, 1).classify(path) == 1)
        os.remove(path)

    def test_random_two(self):
        path = create_num_test(100,100)
        self.assertTrue(KNNForest(path, 4, randint(3,8)/10, 2).classify(path))
        os.remove(path)


if __name__ == '__main__':
    unittest.main()
