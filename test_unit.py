import unittest
import os
import csv

from ID3 import *
from KNN import *
from KNNForest import *
from CostSensitiveKNN import *
from utils import *


""""""""""""""""""""""""""""""""""""""" Useful Methods """""""""""""""""""""""""""""""""""""""


def create_binary_test(num_examples: int, num_features: int, new_path: str = "try"):
    temp_path = "./test_csv/temp.csv"
    actual_path = "./test_csv/"+new_path+".csv"
    df = pd.DataFrame([["M" if randint(0,1) == 1 else "B"] + [randint(0,1) for _ in range(num_features)] for _ in range(num_examples)])
    df.to_csv(temp_path)

    row_count = 0
    with open(temp_path, "r") as source:
        reader = csv.reader(source)
        with open(actual_path, "w", newline='') as result:
            writer = csv.writer(result)
            for row in reader:
                row_count += 1
                for col_index in [0]:
                    del row[col_index]
                writer.writerow(row)

    os.remove(temp_path)
    return actual_path


def create_num_test(num_examples: int, num_features: int, new_path: str = "try"):
    temp_path = "./test_csv/temp.csv"
    actual_path = "./test_csv/"+new_path+".csv"
    df = pd.DataFrame([["M" if randint(0,1) == 1 else "B"] + [randint(0,100) / randint(1,100) for _ in range(num_features)] for _ in range(num_examples)])
    df.to_csv(temp_path)

    row_count = 0
    with open(temp_path, "r") as source:
        reader = csv.reader(source)
        with open(actual_path, "w", newline='') as result:
            writer = csv.writer(result)
            for row in reader:
                row_count += 1
                for col_index in [0]:
                    del row[col_index]
                writer.writerow(row)

    os.remove(temp_path)
    return actual_path


""""""""""""""""""""""""""""""""""""""""" Tests  """""""""""""""""""""""""""""""""""""""""


# class TestID3(unittest.TestCase):
#     def test_continuous(self):
#         test_path = "./test_csv/continuous.csv"
#         self.assertTrue(ID3ContinuousFeatures(test_path).classify(test_path) == 1)
#
#     def test_medium_continuous(self):
#         test_path = "./test_csv/medium_continuous.csv"
#         self.assertTrue(ID3ContinuousFeatures(test_path).classify(test_path) == 1)
#
#     def test_randomly(self):
#         for _ in range(20):
#             test_path = create_num_test(100, 100)
#             self.assertTrue(ID3ContinuousFeatures(test_path).classify(test_path) > 0.95)
#         os.remove("./test_csv/try.csv")
#
#     def test_final_answer(self):
#         self.assertTrue(ID3ContinuousFeatures(TRAIN_PATH).classify(TEST_PATH) == 0.9469026548672567)


# class TestKNN(unittest.TestCase):
#     def test_continuous(self):
#         test_path = "./test_csv/continuous.csv"
#         self.assertTrue(KNN(test_path).classify(test_path) == 1)
#         self.assertFalse(KNN(test_path).classify_and_get_loss(test_path))
#
#     def test_medium_continuous(self):
#         test_path = "./test_csv/medium_continuous.csv"
#         self.assertTrue(KNN(test_path).classify(test_path) == 1)
#         self.assertFalse(KNN(test_path).classify_and_get_loss(test_path))
#
#     def test_randomly(self):
#         for _ in range(10):
#             test_path = create_num_test(100, 100)
#             self.assertTrue(KNN(test_path).classify(test_path) == 1)
#             self.assertFalse(KNN(test_path).classify_and_get_loss(test_path))
#         os.remove("./test_csv/try.csv")
#
#     def test_final_answer(self):
#         self.assertTrue(KNN(TRAIN_PATH).classify(TEST_PATH) == 0.9646017699115044)
#         self.assertFalse(KNN(TRAIN_PATH).classify_and_get_loss(TRAIN_PATH))
#         self.assertFalse(KNN(TEST_PATH).classify_and_get_loss(TEST_PATH))


class TestCostSensitiveKNN(unittest.TestCase):
    def test_final_answer(self):
        self.assertTrue(CostSensitiveKNN(TRAIN_PATH).classify(TEST_PATH) == 0.003539823008849558)

# class TestKNNForest(unittest.TestCase):
#     def test_continuous(self):
#         train_path = "./test_csv/continuous.csv"
#         i = 0
#         for _ in range(1000):
#             a = KNNForest(train_path, 1, 1, 1).classify(train_path)
#             if a != 1:
#                 i += 1
#         self.assertFalse(i)
#
#     def test_final_answer(self):
#         self.assertTrue(KNNForest(TRAIN_PATH, 1, 1, 1).classify(TEST_PATH) == 0.9469026548672567)
#
#     def test_randomly_simple(self):
#         path = create_num_test(100,100)
#         self.assertTrue(KNNForest(path, 1, 1, 1).classify(path) == 1)
#         os.remove(path)
#
#     def test_randomly_complex(self):
#         path = create_num_test(100,100)
#         self.assertTrue(KNNForest(path, 4, randint(3,8)/10, 2).classify(path))
#         os.remove(path)


if __name__ == '__main__':
    unittest.main()
