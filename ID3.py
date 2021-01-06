"""
ID3 Algorithm
"""

from typing import Tuple
from typing import Callable
import pandas as pd
import numpy as np
from math import log2

#TODO: Delete
from random import randint
import csv
import os

Examples = np.array
Features = np.array
Children = list
Classifier = Tuple[int, Children, int]

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"


def get_examples_from_csv(path: str) -> Tuple[Examples, Features]:
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    examples = []
    for row in data_frame.values:
        example = list(row)
        example[0] = 1 if example[0] == "M" else 0
        examples.append(example)

    return np.array(examples), np.array([i+1 for i in range(len(np.transpose(examples))-1)])


""""""""""""""""""""""""""""""""""""""""""" ID3 """""""""""""""""""""""""""""""""""""""""""


class ID3ContinuousFeatures:
    @staticmethod
    def get_classify(train_path) -> Classifier:
        examples, features = get_examples_from_csv(train_path)
        return ID3ContinuousFeatures._tdidt_algorithm(examples, features, ID3ContinuousFeatures._majority_class(examples),
                                                      ID3ContinuousFeatures._max_ig)

    @staticmethod
    def get_accuracy(test_path) -> float:
        pass

    ######### Helper Functions for ID3 Algorithm #########
    @staticmethod
    def _tdidt_algorithm(examples: Examples, features: Features, default: int, select_feature: Callable[[Examples], Tuple[int, Examples, Examples]]) -> Classifier:
        # Empty leaf
        if len(examples) == 0:
            return 0, [], default

        # Consistent node turns leaf
        majority_class = ID3ContinuousFeatures._majority_class(examples)
        if features.size == 0 or ID3ContinuousFeatures._check_consistent_node(examples, majority_class):
            return 0, [], majority_class

        # main decision
        # dynamic_features = ID3ContinuousFeatures._continuous_features(features)
        chosen_feature, class_one, class_two = select_feature(examples)
        assert chosen_feature != 0

        if class_one.size == 0 or class_two.size == 0:  # all the features are same- noise
            return 0, [], majority_class

        # create subtrees fits to the chosen_feature
        subtrees = [ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_one, chosen_feature, 1),
                                                           np.delete(features, chosen_feature-1),
                                                           majority_class,
                                                           select_feature),

                    ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_two, chosen_feature, 1),
                                                           np.delete(features, chosen_feature-1),
                                                           majority_class,
                                                           select_feature)]

        return features[chosen_feature-1], subtrees, majority_class

    @staticmethod
    def _max_ig(examples: Examples) -> Tuple[int, Examples, Examples]:
        father_entropy = ID3ContinuousFeatures._entropy(examples)

        max_class_one = []
        max_class_two = []
        max_ig = -np.inf
        argmax_ig = 0

        for feature in range(1, len(np.transpose(examples))):
            class_true, class_false = ID3ContinuousFeatures._divide_by_feature(examples, feature)
            son1_true_entropy = ID3ContinuousFeatures._entropy(class_true)
            son2_false_entropy = ID3ContinuousFeatures._entropy(class_false)
            ig = father_entropy - (son1_true_entropy * (len(class_true) / len(examples)) + son2_false_entropy * (
                    len(class_false) / len(examples)))

            if max_ig <= ig:
                max_ig = ig
                argmax_ig = feature
                max_class_one = class_true
                max_class_two = class_false

        return argmax_ig, max_class_one, max_class_two

    #@staticmethod
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
        class_true = []
        class_false = []
        example_num = 0
        for i in (examples.transpose())[feature]:
            if i == 1:
                class_true.append(examples[example_num])
            elif i == 0:
                class_false.append(examples[example_num])
            example_num += 1

        return np.array(class_true), np.array(class_false)

    @staticmethod
    def _majority_class(examples: Examples) -> int:
        # assume examples != []
        num_true = np.count_nonzero((examples.transpose())[0])
        if num_true > len(examples) - num_true:
            return 1
        elif num_true < len(examples) - num_true:
            return 0
        else:
            return 1  # TODO: what about tie?

    @staticmethod
    def _check_consistent_node(examples: Examples, c: int) -> bool:
        # assume examples != []
        for example in np.transpose(examples)[0]:
            if example != c:
                return False
        return True

    @staticmethod
    def _entropy(examples: Examples) -> float:
        if examples.size == 0:
            return 0
        num_true = np.count_nonzero((examples.transpose())[0])
        p_true = num_true / len(examples)
        p_false = 1 - p_true

        if p_true == 1.:
            p_false = 1
        elif p_true == 0.:
            p_true = 1

        return -(p_true * log2(p_true) + p_false * log2(p_false))


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def create_test(new_path: str, num_examples: int, num_features: int):
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


def monster_test(repeat: int, examples_num: int, features_num: int):
    try:
        temp_path = "try"
        for successful_test in range(1, repeat+1,):
            ID3ContinuousFeatures.get_classify(create_test(temp_path, examples_num, features_num))
            print(f'{successful_test} successful test was passed')
        os.remove("./test_csv/try.csv")
    except:
        print(f'{successful_test} test threw exception! Tests were failed!')



def test(path):
    classifier = ID3ContinuousFeatures.get_classify(path)
    print(classifier)


def main():
    monster_test(100,1000,1000)
    #test(create_test("try", 7,7))
    #test("./test_csv/binary_all_noise.csv")


if __name__ == "__main__":
    main()


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
