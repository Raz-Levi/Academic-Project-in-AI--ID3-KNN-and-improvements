import unittest
from ID3 import *
from KNN import *
from utils import create_binary_test, create_num_test
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
            self.assertTrue(ID3ContinuousFeatures.classify_without_pruning(test_path, test_path) == 1)
        os.remove("./test_csv/try.csv")

    def test_actual(self):
        self.assertTrue(ID3ContinuousFeatures.classify_without_pruning(TRAIN_PATH, TEST_PATH) == 0.9469026548672567)

    # def test_super_random(self):
    #     test_path = create_num_test(randint(1,1000), randint(1,1000))
    #     self.assertTrue(ID3ContinuousFeatures.classify_without_pruning(test_path, test_path) == 1)
    #     os.remove(test_path)


    # def test_randomly(self):
    #     test_path = create_binary_test(1000, 1000)
    #     self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path) == 1)
    #     os.remove(test_path)
    #
    # def test_monster(self):
    #     for _ in range(3):
    #         test_path = create_binary_test(100, 100)
    #         self.assertTrue(ID3ContinuousFeatures.learn_without_pruning(test_path, test_path) == 1)
    #     os.remove("./test_csv/try.csv")
    #
    # def test_accuracy(self):
    #     print(ID3ContinuousFeatures.learn_without_pruning(create_binary_test(1000, 1000, "train"), create_binary_test(100, 1000, "test")))
    #     os.remove("./test_csv/train.csv")
    #     os.remove("./test_csv/test.csv")
    #
    # def test_pruning(self):
    #     create_binary_test(100, 100, "train")
    #     create_binary_test(10, 100, "test")
    #     print(ID3ContinuousFeatures.learn_without_pruning("./test_csv/train.csv", "./test_csv/test.csv"))
    #     print(ID3ContinuousFeatures.learn_with_pruning("./test_csv/train.csv", "./test_csv/test.csv"))
    #     os.remove("./test_csv/train.csv")
    #     os.remove("./test_csv/test.csv")

    # def test_monster_accuracy(self):
    #     accuracy = []
    #     for _ in range(10):
    #         iteration = ID3ContinuousFeatures.learn_without_pruning(create_test(1000, 1000, "train"), create_test(100, 1000, "test"))
    #         accuracy.append(iteration)
    #         self.assertTrue(0 <= iteration <= 1)
    #     print(max(accuracy))
    #     os.remove("./test_csv/train.csv")
    #     os.remove("./test_csv/test.csv")


# class TestKNN(unittest.TestCase):
#     def test_nominal_csv(self):
#         test_path = "./test_csv/nominal.csv"
#         self.assertTrue(KNN.classify(test_path, test_path) == 1)
#
#     def test_medium_continuous(self):
#         test_path = "./test_csv/medium_continuous.csv"
#         self.assertTrue(learn_k(test_path, test_path) == 1)
#         self.assertTrue(KNN.classify(test_path, test_path))
#
#     def test_continuous(self):
#         test_path = "./test_csv/continuous.csv"
#         self.assertTrue(learn_k(test_path, test_path) == 1)
#         self.assertTrue(KNN.classify(test_path, test_path))
#
#     def test_random(self):
#         test_path = create_num_test(100, 100)
#         self.assertTrue(learn_k(test_path, test_path) == 1)
#         self.assertTrue(KNN.classify(test_path, test_path))
#         os.remove(test_path)

    # def test_knn_monster(self):
    #     for _ in range(1):
    #         test_path = create_num_test(100, 1000)
    #         self.assertTrue(learn_k(test_path, test_path, False) == 1)
    #
    #         self.assertTrue(KNN.learn(test_path, test_path, False))
    #     os.remove("./test_csv/try.csv")

    # def test_actual_data_train(self):
    #     test_path = "./train.csv"
    #     self.assertTrue(learn_k(test_path, test_path) == 1)
    #     self.assertTrue(KNN.classify(test_path, test_path))
    #
    # def test_actual_data(self):
    #     test_path = "./test.csv"
    #     self.assertTrue(learn_k(test_path, test_path) == 1)
    #     self.assertTrue(KNN.classify(test_path, test_path))
    #
    # def test_actual_accuracy(self):
    #     train_path = "./train.csv"
    #     test_path = "./test.csv"
    #     self.assertTrue(KNN.classify(train_path, test_path) == 0.9734513274336283)
    #
    # def test_loss(self):
    #     train_path = "./train.csv"
    #     test_path = "./test.csv"
    #     self.assertTrue(KNN.classify_and_get_loss(train_path, test_path) == 0.018584070796460177)


if __name__ == '__main__':
    unittest.main()
