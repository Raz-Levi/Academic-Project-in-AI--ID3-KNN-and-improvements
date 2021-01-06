"""
ID3 Algorithm
"""
import numpy as np
from utils import get_examples_from_csv
from math import log2
from random import randint

from typing import Tuple
from typing import Callable

Examples = np.array
Features = np.array
Children = list
Classifier = Tuple[int, Children, int]

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

""""""""""""""""""""""""""""""""""""""""""" ID3 """""""""""""""""""""""""""""""""""""""""""


class ID3ContinuousFeatures:
    @staticmethod
    def get_classifier(train_path: str) -> Classifier:
        examples, features = get_examples_from_csv(train_path)
        return ID3ContinuousFeatures._tdidt_algorithm(examples, features,
                                                      ID3ContinuousFeatures._majority_class(examples),
                                                      ID3ContinuousFeatures._max_ig)

    @staticmethod
    def get_accuracy(classifier: Classifier, test_path: str) -> float:
        examples, _ = get_examples_from_csv(test_path)
        true_pos, true_neg = 0, 0
        for example in examples:
            example_result = ID3ContinuousFeatures._test_example(classifier, example)
            if example_result == 1 and example[0] == 1:
                true_pos += 1
            elif example_result == 0 and example[0] == 0:
                true_neg += 1

        return (true_pos + true_neg) / len(examples)

    @staticmethod
    def learn(train_path: str, test_path: str) -> float:
        return ID3ContinuousFeatures.get_accuracy(ID3ContinuousFeatures.get_classifier(train_path), test_path)

    ######### Helper Functions for ID3 Algorithm #########
    @staticmethod
    def _tdidt_algorithm(examples: Examples, features: Features, default: int,
                         select_feature: Callable[[Examples], Tuple[int, Examples, Examples]],
                         M: int = 1) -> Classifier:
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
        assert chosen_feature != 0  # TODO: Delete

        if class_one.size == 0 or class_two.size == 0:  # all the features are same- noise
            return 0, [], majority_class

        # create subtrees fits to the chosen_feature
        subtrees = [ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_one, chosen_feature, 1),
                                                           np.delete(features, chosen_feature - 1),
                                                           majority_class,
                                                           select_feature),

                    ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_two, chosen_feature, 1),
                                                           np.delete(features, chosen_feature - 1),
                                                           majority_class,
                                                           select_feature)]

        return features[chosen_feature - 1], subtrees, majority_class

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

    # @staticmethod
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
            return randint(0, 1)

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

    @staticmethod
    def _test_example(classifier: Classifier, example: Examples) -> int:
        # assume classifier seems like: ( chosen_feature, [children: true, false], classification)
        if len(classifier[1]) == 0:  # if children == []: take classification
            return classifier[2]

        if example[classifier[0]]:
            return ID3ContinuousFeatures._test_example(classifier[1][0], example)
        return ID3ContinuousFeatures._test_example(classifier[1][1], example)


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def main():
    pass


if __name__ == "__main__":
    main()
