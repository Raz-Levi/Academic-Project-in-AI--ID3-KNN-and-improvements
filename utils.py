"""
Utils For Project
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""


import pandas as pd
import numpy as np
from random import randint, shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from bisect import insort

from typing import Tuple
from typing import Callable


""""""""""""""""""""""""""""""""""""""""" Definitions """""""""""""""""""""""""""""""""""""""""


Examples = np.array
Features = np.array
Forest = np.array
Centroid = np.array
Classifier = Tuple[Tuple[int, float], Examples, Examples]

TRAIN_PATH = "./train.csv"
TEST_PATH = "./test.csv"
POSITIVE_SIGN = "M"
M_VALUES = (1, 3, 5, 7, 9)
N_SPLIT = 5
SHUFFLE = True
RANDOM_STATE = 316579275
NUM_FOR_CHOOSE = 5

""""""""""""""""""""""""""""""""""""""" Useful Classes """""""""""""""""""""""""""""""""""""""


class CommitteeWrapper(object):
    def __init__(self, classification: int, distance: float):
        self.classification_or_classifier = classification
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __eq__(self, other: int):
        return self.classification_or_classifier == other


""""""""""""""""""""""""""""""""""""""" Useful Methods """""""""""""""""""""""""""""""""""""""


def get_full_examples_from_csv(path: str) -> Examples:
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
