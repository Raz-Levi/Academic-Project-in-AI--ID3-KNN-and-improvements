from utils import *
import ID3
from dataclasses import dataclass


@dataclass
class Tree:
    classifier: Classifier
    centroid: Centroid


class KNNForest(object):
    def _create_forest(self, examples: Examples, num_trees: int, p: float) -> Forest:
        forest = []
        for _ in range(num_trees):
            chosen_examples = np.take(examples, sample(examples, int(len(examples)*p)), 0)
            assert len(chosen_examples) == int(len(examples)*p)  # TODO: Delete!
            forest.append(Tree(
                classifier=ID3.ID3ContinuousFeatures.get_classifier(chosen_examples),
                centroid=np.average(chosen_examples, axis=0)))

        return np.array(forest)

    def _classify_one(self, test_example: Examples, forest: Forest, num_chosen_trees: int) -> int:

        class DistanceWrapper(object):
            def __init__(self, classifier: int, distance: float):
                self.classifier = classifier
                self.distance = distance

            def __lt__(self, other):
                return self.distance < other.distance

        distances = []
        for tree in forest:
            insort(distances, DistanceWrapper(tree.classifier, euclidean_distance(tree.centroid, test_example)))

        trees_num = 0
        vote_true = 0
        for classifier in distances:
            if trees_num >= num_chosen_trees:
                break
            if ID3.ID3ContinuousFeatures.classify_one(classifier.classifier, test_example) == 1:
                vote_true += 1
            trees_num += 1
        return 1 if vote_true >= min(num_chosen_trees, len(distances)) - vote_true else 0


def main():
    pass


if __name__ == '__main__':
    main()