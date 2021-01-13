"""
Abstract Class for Learning Algorithm
"""
import abc
from utils import get_full_examples_from_csv, Examples

""""""""""""""""""""""""""""""""""""""""""" LearningAlgorithm """""""""""""""""""""""""""""""""""""""""""


class LearningAlgorithm(object):
    def __init__(self, train_path: str):
        self._train_examples = get_full_examples_from_csv(train_path)

    @abc.abstractmethod
    def classify(self, test_path: str) -> float:
        ...

    @abc.abstractmethod
    def _get_accuracy(self, test_examples: Examples) -> float:
        ...

    @abc.abstractmethod
    def _classify_one(self, test_example: Examples) -> int:
        ...
