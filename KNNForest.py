"""
KNNForest Algorithm
"""
from LearningAlgorithm import LearningAlgorithm
from utils import *
import ID3
from dataclasses import dataclass

BEST_N = 4
BEST_P = 0.3
BEST_K = 3

""""""""""""""""""""""""""""""""""""""""""" KNNForest """""""""""""""""""""""""""""""""""""""""""


@dataclass
class Tree:
    classifier: Classifier
    centroid: Centroid


class KNNForest(LearningAlgorithm):
    def __init__(self, train_path: str, num_trees: int = BEST_N, p: float = BEST_P, num_chosen_trees: int = BEST_K):
        super().__init__(train_path)
        self._num_trees = num_trees
        self._p = p
        self._num_chosen_trees = num_chosen_trees
        self._forest = self._create_forest()

    def classify(self, test_path: str) -> float:
        return self._get_accuracy(get_generator_examples_from_csv(test_path))

    ######### Helper Functions for KNNForest Algorithm #########

    def _create_forest(self) -> Forest:
        forest = []
        indexes = [i for i in range(len(self._train_examples))]
        for _ in range(self._num_trees):
            shuffle(indexes)
            chosen_examples = np.take(self._train_examples, indexes[0:int((len(self._train_examples)*self._p))], 0)
            assert len(chosen_examples) == int(len(self._train_examples)*self._p)  # TODO: Delete!
            forest.append(Tree(
                classifier=ID3.ID3ContinuousFeatures.get_classifier(chosen_examples),
                centroid=np.average(np.delete(chosen_examples, 0, 1), axis=0)))

        return np.array(forest)

    def _classify_one(self, test_example: Examples) -> int:

        class DistanceWrapper(object):
            def __init__(self, _classifier: int, _distance: float):
                self.classifier = _classifier
                self.distance = _distance

            def __lt__(self, other):
                return self.distance < other.distance

        distances = []
        for tree in self._forest:
            insort(distances, DistanceWrapper(tree.classifier, euclidean_distance(tree.centroid, test_example[1:])))

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
    print(KNNForest(TRAIN_PATH).classify(TEST_PATH))


if __name__ == '__main__':
    main()
