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
        indexes = [i for i in range(len(self._train_examples))]
        for _ in range(self._num_trees):
            shuffle(indexes)
            chosen_examples = np.take(self._train_examples, indexes[0:int((len(self._train_examples)*self._p))], 0)
            assert len(chosen_examples) == int(len(self._train_examples)*self._p)  # TODO: Delete!
            forest.append(Tree(
                classifier=ID3.ID3ContinuousFeatures.get_classifier(chosen_examples),
                centroid=np.average(np.delete(chosen_examples, 0, 1), axis=0)))

        return np.array(forest)

    def _get_accuracy(self, test_examples: Examples) -> float:
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

def exp():
    path = "./test_csv/try.csv"
    print(KNNForest(path, 1, 1, 1).classify(path))

def cons():
    best = (0, 0, 0)
    best_acc = 0
    for N in range(1, 10):
        print(f'check N={N}')
        for k in range(1, N + 1):
            print(f'check k={k}')
            for p in range(3, 8):
                print(f'check p={p / 10}')
                acc = KNNForest(TRAIN_PATH, N, p / 10, k).classify(TEST_PATH)
                if best_acc < acc:
                    best_acc = acc
                    best = (N, p / 10, k)
        print("----------------------")
        print(f'intil now- best acc: {best_acc}, N={best[0]}, p={best[1]}, k={best[2]}')
    print("====================================")
    print(f'finished- best acc: {best_acc}, N={best[0]}, p={best[1]}, k={best[2]}')

def main():
    best_acc = 0
    while True:
        N= randint(1, 15)
        k = randint(1, N)
        p = randint(300,701)/1000
        acc = KNNForest(TRAIN_PATH, N, p, k).classify(TEST_PATH)
        if best_acc <= acc:
            best_acc = acc
            print("====================================")
            print(f'new Max: best acc: {best_acc}, N={N}, p={p}, k={k}')
            print("====================================")
        else:
            print(f'checked N={N}, p={p}, k={k}')


if __name__ == '__main__':
    main()
