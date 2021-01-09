"""
Utils For Project
"""
#TODO: Delete
import csv
import os

import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from typing import Tuple
from typing import Callable
Examples = np.array
Features = np.array
Children = list
Classifier = Tuple[int, Children, int]

TRAIN_PATH = "./train.csv"
TEST_PATH = "./test.csv"
N_SPLIT = 5
SHUFFLE = True
RANDOM_STATE = 316579275
NUM_FOR_CHOOSE = 5


def get_full_examples_from_csv(path: str) -> Examples:  # TODO: Remove features
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    examples = []
    for row in data_frame.values:
        example = list(row)
        example[0] = 1 if example[0] == "M" else 0
        examples.append(example)
    return np.array(examples)


def get_generator_examples_from_csv(path: str) -> Examples:
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    for row in data_frame.values:
        example = list(row)
        example[0] = 1 if example[0] == "M" else 0
        yield example


def print_graph(values: list, accuracy: list, char: str):
    plt.plot(values, accuracy)
    plt.ylabel('Average accuracy')
    plt.xlabel(f'{char} values')
    plt.show()


""""""""""""""""""""""""""""""""""""""""""" Tests """""""""""""""""""""""""""""""""""""""""""


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


# def get_full_examples_from_csv(path: str, get_features: bool = False) -> Tuple[Examples, Features]:  # TODO: Remove features
#     data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
#     examples = []
#     for row in data_frame.values:
#         example = list(row)
#         example[0] = 1 if example[0] == "M" else 0
#         examples.append(example)
#     examples = np.array(examples)
#     if get_features:
#         return examples, np.array([i for i in range(1, len(data_frame.columns))])
#     return examples, ()

# def monster_test(repeat: int, examples_num: int, features_num: int):
#     failed_test = 0
#     try:
#         temp_path = "try"
#         for successful_test in range(1, repeat+1,):
#             ID3ContinuousFeatures._get_classifier(create_test(temp_path, examples_num, features_num))
#             print(f'{successful_test} successful test was passed')
#             failed_test +=1
#         os.remove("./test_csv/try.csv")
#     except:
#         print(f'{failed_test} test threw exception! Tests were failed!')
#
#
# def classifier_test(path):
#     classifier = ID3ContinuousFeatures._get_classifier(path)
#     print(classifier)
#
#
# def learn_test(path):
#     actual_path = "./test_csv/" + path + ".csv"
#     learn_result = ID3ContinuousFeatures.learn_without_pruning(actual_path, actual_path) == 1
#     print(learn_result)
#     assert learn_result == 1
#
#
# def accuracy_test():
#     print(ID3ContinuousFeatures.learn_without_pruning("./test_csv/train.csv", "./test_csv/test.csv"))
#
#
# def random_accuracy_test():
#     print(ID3ContinuousFeatures.learn_without_pruning(create_test(1000, 1000, "train"), create_test(100, 1000, "test")))


# @staticmethod
# def _binary_tdidt_algorithm(examples: Examples, features: Features, default: int,
#                      select_feature: Callable[[Examples], Tuple[int, Examples, Examples]],
#                      M: int) -> Classifier:
#     # Empty leaf
#     if len(examples) == 0:
#         return 0, [], default
#
#     # Consistent node turns leaf
#     majority_class = ID3ContinuousFeatures._majority_class(examples)
#     if len(examples) <= M or features.size == 0 or ID3ContinuousFeatures._check_consistent_node(examples,
#                                                                                                 majority_class):
#         return 0, [], majority_class
#
#     # main decision
#     # dynamic_features = ID3ContinuousFeatures._continuous_features(features)
#     chosen_feature, class_one, class_two = select_feature(examples)
#
#     if class_one.size == 0 or class_two.size == 0:  # all the features are same- noise
#         return 0, [], majority_class
#
#     # create subtrees fits to the chosen_feature
#     subtrees = [ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_one, chosen_feature, 1),
#                                                        np.delete(features, chosen_feature - 1),
#                                                        majority_class,
#                                                        select_feature, M),
#
#                 ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_two, chosen_feature, 1),
#                                                        np.delete(features, chosen_feature - 1),
#                                                        majority_class,
#                                                        select_feature, M)]
#
#     return features[chosen_feature - 1], subtrees, majority_class