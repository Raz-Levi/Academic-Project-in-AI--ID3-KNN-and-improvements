"""
Utils For Project
"""
#TODO: Delete
import csv
import os

import pandas as pd
import numpy as np
from random import randint, shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from bisect import insort

from typing import Tuple
from typing import Callable

Examples = np.array
Features = np.array
Forest = np.array
Centroid = np.array
Children = list
Classifier = Tuple[Tuple[int, float], Examples, Examples]

TRAIN_PATH = "./train.csv"
TEST_PATH = "./test.csv"
N_SPLIT = 5
SHUFFLE = True
RANDOM_STATE = 123456789  # TODO: change to my ID!
NUM_FOR_CHOOSE = 5
POSITIVE_SIGN = "M"


def get_full_examples_from_csv(path: str) -> Examples:  # TODO: Remove features
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    examples = []
    for row in data_frame.values:
        example = list(row)
        example[0] = 1 if example[0] == POSITIVE_SIGN else 0
        examples.append(example)
    return np.array(examples)


def get_generator_examples_from_csv(path: str) -> Examples:
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    for row in data_frame.values:
        example = list(row)
        example[0] = 1 if example[0] == "M" else 0
        yield example


def print_graph(values: list, accuracy: list, char: str):
    plt.plot(values, accuracy, 'ro')
    plt.ylabel('Average accuracy')
    plt.xlabel(f'{char} values')
    plt.show()


def euclidean_distance(example_one: Examples, example_two: Examples) -> float:
    # assume len(example_one) == len(example_two)
    distance = 0
    is_feature = False
    for feature_one, feature_two in zip(example_one, example_two):  # the first cell in example is not a feature
        if not is_feature:
            is_feature = True
            continue
        distance += (feature_one - feature_two) ** 2

    return distance ** 0.5


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