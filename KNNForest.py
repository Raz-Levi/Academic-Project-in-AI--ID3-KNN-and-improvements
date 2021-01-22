"""
KNNForest Algorithm
"""
from LearningAlgorithm import LearningAlgorithm
from utils import *
from ID3 import ID3ContinuousFeatures
from dataclasses import dataclass

BEST_N = 4
BEST_P = 0.3
BEST_K = 3

""""""""""""""""""""""""""""""""""""""""""" KNNForest """""""""""""""""""""""""""""""""""""""""""


@dataclass
class Tree:
    classifier: ID3ContinuousFeatures
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

    def classify_one(self, test_example: Examples) -> int:
        committee = []
        for tree in self._forest:
            insort(committee, CommitteeWrapper(tree.classifier, euclidean_distance(tree.centroid, test_example[1:])))

        vote_for, vote_against = 0, 0
        for classifier in committee:
            if vote_for + vote_against >= self._num_chosen_trees:
                break
            if classifier.classification_or_classifier.classify_one(test_example) == 1:
                vote_for += 1
            else:
                vote_against += 1
        return 1 if vote_for >= vote_against else 0

    ######### Helper Functions for KNNForest Algorithm #########

    def _create_forest(self) -> Forest:
        forest = []
        indexes = [i for i in range(len(self._train_examples))]
        for _ in range(self._num_trees):
            shuffle(indexes)
            chosen_examples = np.take(self._train_examples, indexes[0:int((len(self._train_examples) * self._p))], 0)
            assert len(chosen_examples) == int(len(self._train_examples) * self._p)  # TODO: Delete!
            forest.append(
                Tree(
                    classifier=ID3ContinuousFeatures(chosen_examples),
                    centroid=np.average(np.delete(chosen_examples, 0, 1), axis=0)
                )
            )

        return np.array(forest)


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def experiment_loop():
    train_examples = my(TRAIN_PATH)
    folds = KFold(n_splits=N_SPLIT, shuffle=SHUFFLE, random_state=RANDOM_STATE)
    n_values = [i for i in range(1, 16)]
    avg_accuracy = []

    for n in n_values:
        for k in range(1,n):
            for p in (i/10 for i in range(3,8)):
                accuracy = 0
                for train_fold, test_fold in folds.split(train_examples):
                    accuracy += KNNForest(np.take(train_examples, train_fold, 0), n, p, k).classify(
                        np.take(train_examples, test_fold, 0))
                avg_accuracy.append((k, p, accuracy / N_SPLIT))
                print(f'finished check n={n}, k={k}, p={p}, got:\n{avg_accuracy}\n')

    return avg_accuracy


def experiment_random(train_path):
    train_examples = my(train_path)
    folds = KFold(n_splits=N_SPLIT, shuffle=SHUFFLE, random_state=RANDOM_STATE)
    avg_accuracy = []

    while True:
        accuracy = 0
        n= randint(1,15)
        k = randint(1, n)
        p = randint(3000, 8000) / 1000
        for train_fold, test_fold in folds.split(train_examples):
            accuracy += KNNForest(np.take(train_examples, train_fold, 0), n, p, k).classify(
                np.take(train_examples, test_fold, 0))
        avg_accuracy.append((k, p, accuracy / N_SPLIT))
        print(f'finished check n={n}, k={k}, p={p}, got:\n{avg_accuracy}\n')

def main():
    print(KNNForest(TRAIN_PATH).classify(TEST_PATH))


if __name__ == '__main__':
    # main()
    experiment_loop()