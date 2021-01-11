"""
KNNForest Algorithm
"""
from utils import *
import ID3
from dataclasses import dataclass


""""""""""""""""""""""""""""""""""""""""""" KNNForest """""""""""""""""""""""""""""""""""""""""""


@dataclass
class Tree:
    classifier: Classifier
    centroid: Centroid


class KNNForest(object):
    def __init__(self, train_path: str, num_trees: int, p: float, num_chosen_trees: int):
        self._train_examples = get_full_examples_from_csv(train_path)
        self._num_trees = num_trees
        self._p = p
        self._num_chosen_trees = num_chosen_trees
        self._forest = self._create_forest()

    def classify(self, test_path: str) -> float:
        return self._get_accuracy(get_generator_examples_from_csv(test_path))

    ######### Helper Functions for KNNForest Algorithm #########

    def _create_forest(self) -> Forest:
        forest = []
        for _ in range(self._num_trees):
            chosen_examples = np.take(self._train_examples, sample(self._train_examples, int(len(self._train_examples)*self._p)), 0)
            assert len(chosen_examples) == int(len(self._train_examples)*self._p)  # TODO: Delete!
            forest.append(Tree(
                classifier=ID3.ID3ContinuousFeatures.get_classifier(chosen_examples),
                centroid=np.average(chosen_examples, axis=0)))

        return np.array(forest)

    def _get_accuracy(self, test_examples: Generator[Examples]) -> float:
        classify_correct, test_examples_amount = 0, 0
        for example in test_examples:
            example_result = self._classify_one(example)
            if example_result == example[0]:
                classify_correct += 1
            test_examples_amount += 1

        return classify_correct / test_examples_amount

    def _classify_one(self, test_example: Examples) -> int:

        class DistanceWrapper(object):
            def __init__(self, _classifier: int, _distance: float):
                self.classifier = _classifier
                self.distance = _distance

            def __lt__(self, other):
                return self.distance < other.distance

        distances = []
        for tree in self._forest:
            insort(distances, DistanceWrapper(tree.classifier, euclidean_distance(tree.centroid, test_example)))

        trees_num = 0
        vote_true = 0
        for classifier in distances:
            if trees_num >= self._num_chosen_trees:
                break
            if ID3.ID3ContinuousFeatures.classify_one(classifier.classifier, test_example) == 1:
                vote_true += 1
            trees_num += 1
        return 1 if vote_true >= min(self._num_chosen_trees, len(distances)) - vote_true else 0


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def main():
    pass


if __name__ == '__main__':
    main()
