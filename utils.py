"""
Utils For Project
"""
#TODO: Delete
from random import randint
import csv
import os
from ID3 import *

import pandas as pd
import numpy as np

from typing import Tuple

Examples = np.array
Features = np.array


def get_full_examples_from_csv(path: str) -> Tuple[Examples, Features]:
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    examples = []
    for row in data_frame.values:
        example = list(row)
        example[0] = 1 if example[0] == "M" else 0
        examples.append(example)

    return np.array(examples), np.array([i for i in range(1, len(data_frame.columns)+1)])


def get_generator_examples_from_csv(path: str) -> Tuple[Examples, Features]:
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    for row in data_frame.values:
        example = list(row)
        example[0] = 1 if example[0] == "M" else 0
        yield example


""""""""""""""""""""""""""""""""""""""""""" Tests """""""""""""""""""""""""""""""""""""""""""


def create_test(num_examples: int, num_features: int, new_path: str = "try"):
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
    failed_test = 0
    try:
        temp_path = "try"
        for successful_test in range(1, repeat+1,):
            ID3ContinuousFeatures.get_classifier(create_test(temp_path, examples_num, features_num))
            print(f'{successful_test} successful test was passed')
            failed_test +=1
        os.remove("./test_csv/try.csv")
    except:
        print(f'{failed_test} test threw exception! Tests were failed!')


def classifier_test(path):
    classifier = ID3ContinuousFeatures.get_classifier(path)
    print(classifier)


def learn_test(path):
    actual_path = "./test_csv/" + path + ".csv"
    learn_result = ID3ContinuousFeatures.learn_without_pruning(actual_path, actual_path) == 1
    print(learn_result)
    assert learn_result == 1


def accuracy_test():
    print(ID3ContinuousFeatures.learn_without_pruning("./test_csv/train.csv", "./test_csv/test.csv"))


def random_accuracy_test():
    print(ID3ContinuousFeatures.learn_without_pruning(create_test(1000, 1000, "train"), create_test(100, 1000, "test")))