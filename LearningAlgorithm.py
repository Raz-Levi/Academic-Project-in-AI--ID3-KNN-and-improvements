"""
Abstract Class for Learning Algorithm
"""
import abc
from utils import get_full_examples_from_csv, Examples

""""""""""""""""""""""""""""""""""""""""""" LearningAlgorithm """""""""""""""""""""""""""""""""""""""""""


class LearningAlgorithm(abc.ABC):
    def __init__(self, train_path: str):
        self._train_examples = get_full_examples_from_csv(train_path)

    @abc.abstractmethod
    def classify(self, test_path: str) -> float:
        ...

    def _get_accuracy(self, test_examples: Examples) -> float:
        classify_correct, test_examples_amount = 0, 0
        for example in test_examples:
            example_result = self._classify_one(example)
            if example_result == example[0]:
                classify_correct += 1
            test_examples_amount += 1

        return classify_correct / test_examples_amount

    @abc.abstractmethod
    def _classify_one(self, test_example: Examples) -> int:
        ...
