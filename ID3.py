"""
ID3 Algorithm
"""

from typing import Tuple
import pandas as pd
import numpy as np

Features = np.array
Examples = np.array

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"


def get_examples_and_features_from_csv(path: str) -> Tuple[Examples, Features]:
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")

    feature_num = 0
    examples = []
    features = []
    for row_in_df in data_frame.values:
        row = list(row_in_df)
        examples.append((feature_num, row[0]))
        del row[0]
        features.append((feature_num,row))
        feature_num += 1

    return np.array(examples), np.array(features)


""""""""""""""""""""""""""""""""""""""""""" ID3 """""""""""""""""""""""""""""""""""""""""""


class ID3:
    @staticmethod
    def get_classify(train_path):
        examples, features = get_examples_and_features_from_csv(train_path)
        return ID3._id3_algorithm_all_features_continuous(examples, features)

    @staticmethod
    def get_accuracy(test_path) -> float:
        pass

    ######### Helper Functions for ID3 Algorithm #########
    @staticmethod
    def _id3_algorithm_all_features_continuous(examples: Examples, features: Features):
        def _max_ig(examples: Examples, features: Features):

            for feature in features.transpose():


        return ID3._tdidt_algorithm_all_features_continuous(examples, features, ID3._majority_class(examples), _max_ig)

    @staticmethod
    def _tdidt_algorithm_all_features_continuous(examples: Examples, features: Features, default: str, select_feature):
        # Empty leaf
        if not examples:
            return None, [], default

        # Consistent node turns leaf
        majority_class = ID3._majority_class(examples)
        if not features or ID3._check_consistent_node(examples, majority_class):
            return None, [], majority_class  # TODO: what about tie?

        # main decision
        dynamic_features = ID3._continuous_features(features)
        chosen_feature = select_feature(examples, dynamic_features)
        features.remove(chosen_feature)  # TODO

        # create subtrees fits to the chosen_feature
        feature_value_one = []
        feature_value_two = []
        for example in examples:



        subtrees = [ID3._tdidt_algorithm_all_features_continuous(, features, majority_class, select_feature) for v in [0,1]]
        # return chosen_feature, subtrees, majority_class

    @staticmethod
    def _continuous_features(features: Features) -> Features:
        print(features)
        continuous_features = []

        for feature in features.transpose():
            features_values = sorted(feature)

            parts = []
            for part in range(len(features_values)-1):
                parts.append((features_values[part]+features_values[part+1])//2)

            continuous_feature = []
            for part in parts:
                for old_feature in features_values:
                    continuous_feature.append(0 if old_feature <= part else 1)

            continuous_features.append(continuous_feature)

        return np.array(continuous_features).transpose()

    ######### Helper Functions in class #########
    # return the list: [number of class1, number of class2, majority class]
    @staticmethod
    def _classes_size_and_majority(examples: Examples) -> list:
        if not examples:
            return [0,0,""]

        class_one = [0,examples[0][1]]
        class_two = [0,""]
        for i in range(1, len(examples)):
            if examples[i,1] == class_one:
                class_one[0] += 1
            else:
                class_two[1] = examples[i][1]
                class_two[0] -= 1

        return [class_one[0], class_two[0], class_one[1] if class_one[0] >= class_two[0] else class_two[1]]

    @staticmethod
    def _majority_class(examples: Examples) -> str:
        return ID3._classes_size_and_majority(examples)[2]

    @staticmethod
    def _check_consistent_node(examples: Examples, c: str) -> bool:
        for example in examples:
            if example[1] != c:
                return False
        return True

    @staticmethod
    def _entropy(examples: Examples) -> float:
        for example in examples:


    @staticmethod
    def _ig(examples: Examples) -> float:



""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""

# TODO: remove
def test_read_csv():
    examples, features = get_examples_and_features_from_csv("try.csv")
    assert len(examples) == len(features)
    print(examples)
    print(features[0])

def test_continuous_features():
    _, features = get_examples_and_features_from_csv("try2.csv")
    print(ID3._continuous_features(features))

def main():
    test_continuous_features()


if __name__ == "__main__":
    main()


