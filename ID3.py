"""
ID3 Algorithm
"""

from typing import Tuple
import pandas as pd
import numpy as np
from math import log2

Examples = np.array
Feature = int
Children = list
Classifier = Tuple[Feature, Children, int]

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"


# TODO: Delete
def get_examples_and_features_from_csv(path: str) -> Tuple[Examples, Examples]:
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")

    feature_num = 0
    examples = []
    features = []
    for row_in_df in data_frame.values:
        row = list(row_in_df)
        examples.append((feature_num, row[0]))
        del row[0]
        features.append((feature_num, row))
        feature_num += 1

    return np.array(examples), np.array(features)


def get_examples_from_csv(path: str) -> Examples:
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    examples = []
    for row in data_frame.values:
        example = list(row)
        example[0] = 1 if example[0] == "M" else 0
        examples.append(example)
    return np.array(examples)


""""""""""""""""""""""""""""""""""""""""""" ID3 """""""""""""""""""""""""""""""""""""""""""


class ID3ContinuousFeatures:
    @staticmethod
    def get_classify(train_path) -> Classifier:
        examples = get_examples_from_csv(train_path)
        return ID3ContinuousFeatures._tdidt_algorithm(examples, ID3ContinuousFeatures._majority_class(examples),
                                                      ID3ContinuousFeatures._max_ig)

    @staticmethod
    def get_accuracy(test_path) -> float:
        pass

    ######### Helper Functions for ID3 Algorithm #########
    @staticmethod
    def _tdidt_algorithm(examples: Examples, default: int, select_feature) -> Classifier:
        # Empty leaf
        if not examples:
            return 0, [], default

        # Consistent node turns leaf
        majority_class = ID3ContinuousFeatures._majority_class(examples)
        if len(examples[0]) == 1 or ID3ContinuousFeatures._check_consistent_node(examples, majority_class):
            return 0, [], majority_class

        # main decision
        # dynamic_features = ID3ContinuousFeatures._continuous_features(features)
        chosen_feature, class_one, class_two = select_feature(examples)

        # create subtrees fits to the chosen_feature
        subtrees = [ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_one, chosen_feature, 1), majority_class,
                                                           select_feature),
                    ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_two, chosen_feature, 1), majority_class,
                                                           select_feature)]
        return chosen_feature, subtrees, majority_class

    @staticmethod
    def _max_ig(examples: Examples) -> Tuple[Feature, Examples, Examples]:
        father_entropy = ID3ContinuousFeatures._entropy(examples)

        max_class_one = []
        max_class_two = []
        max_feature_entropy = -np.inf
        max_feature = 0

        for feature in range(1, len(examples)):
            class_one, class_two = ID3ContinuousFeatures._divide_by_feature(examples, feature)
            son1_entropy = ID3ContinuousFeatures._entropy(class_one)
            son2_entropy = ID3ContinuousFeatures._entropy(class_two)
            ig = father_entropy - (son1_entropy * (len(class_one) / len(examples)) + son2_entropy * (
                    len(class_two) / len(examples)))

            if max_feature_entropy <= ig:
                max_feature_entropy = ig
                max_feature = feature
                max_class_one = class_one
                max_class_two = class_two

        return max_feature, max_class_one, max_class_two

    @staticmethod
    # def _continuous_features(features: Features) -> Features:
    #     print(features)
    #     continuous_features = []
    #
    #     for feature in features.transpose():
    #         features_values = sorted(feature)
    #
    #         parts = []
    #         for part in range(len(features_values)-1):
    #             parts.append((features_values[part]+features_values[part+1])//2)
    #
    #         continuous_feature = []
    #         for part in parts:
    #             for old_feature in features_values:
    #                 continuous_feature.append(0 if old_feature <= part else 1)
    #
    #         continuous_features.append(continuous_feature)
    #
    #     return np.array(continuous_features).transpose()

    ######### Helper Functions in class #########
    @staticmethod
    def _divide_by_feature(examples: Examples, feature: int) -> Tuple[Examples, Examples]:
        class_one = []
        class_two = []
        example_num = 0
        for i in (examples.transpose())[feature]:
            if i == 1:
                class_one.append(examples[example_num])
            elif i == 0:
                class_two.append(examples[example_num])
            example_num += 1

        return np.array(class_one), np.array(class_two)

    @staticmethod
    def _majority_class(examples: Examples) -> int:
        # assume examples != []
        num_true = np.count_nonzero((examples.transpose())[0])
        if num_true > len(examples) - num_true:
            return 1
        elif num_true < len(examples) - num_true:
            return 0
        else:
            return -1  # TODO: what about tie?

    @staticmethod
    def _check_consistent_node(examples: Examples, c: int) -> bool:
        # assume examples != []
        for example in np.transpose(examples)[0]:
            if example != c:
                return False
        return True

    @staticmethod
    def _entropy(examples: Examples) -> float:
        num_true = np.count_nonzero((examples.transpose())[0])
        p_true = num_true / len(examples)
        p_false = (len(examples) - num_true) / len(examples)

        return -(p_true * log2(p_true) + p_false * log2(p_false))


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


# TODO: remove
def test_read_csv():
    examples, features = get_examples_from_csv("try.csv")
    assert len(examples) == len(features)
    print(examples)
    print(features[0])


# def test_continuous_features():
#     _, features = get_examples_from_csv("try2.csv")
#     print(ID3ContinuousFeatures._continuous_features(features))

def main():
    test_read_csv()


if __name__ == "__main__":
    main()
