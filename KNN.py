"""
KNN Algorithm
"""
from LearningAlgorithm import LearningAlgorithm
from utils import get_full_examples_from_csv, Examples, CommitteeWrapper, euclidean_distance, TRAIN_PATH, TEST_PATH, np, insort

BEST_K = 1

""""""""""""""""""""""""""""""""""""""""""" KNN """""""""""""""""""""""""""""""""""""""""""


class KNN(LearningAlgorithm):
    def __init__(self, train_path: str, k: int = BEST_K):
        super().__init__(train_path)
        self._min_max_values = self._train_minmax_normalize()
        self._k = k

    def classify(self, test_path: str) -> float:
        test_examples = get_full_examples_from_csv(test_path)
        self._test_minmax_normalize(test_examples)
        return self._get_accuracy(test_examples)

    def classify_and_get_loss(self, test_path) -> float:
        test_examples = get_full_examples_from_csv(test_path)
        self._test_minmax_normalize(test_examples)
        return self._get_loss(test_examples)

    def classify_one(self, test_example: Examples) -> int:
        committee = []
        for train_example in self._train_examples:
            insort(committee, CommitteeWrapper(train_example[0], euclidean_distance(test_example, train_example)))

        vote_for, vote_against = 0, 0
        for vote in committee:
            if vote_for + vote_against >= self._k:
                break
            if vote == 1:
                vote_for += 1
            else:
                vote_against += 1
        return 1 if vote_for >= vote_against else 0

    ######### Helper Functions for KNN Algorithm #########

    def _get_loss(self, test_examples: Examples) -> float:
        fp, fn = 0, 0
        for example in test_examples:
            example_result = self.classify_one(example)
            if example_result == 1 and example[0] == 0:
                fp += 1
            elif example_result == 0 and example[0] == 1:
                fn += 1

        return (0.1 * fp + fn) / len(test_examples)

    def _train_minmax_normalize(self) -> list:
        min_max_values = []
        is_classifier = True

        for example in self._train_examples.transpose():
            if is_classifier:
                is_classifier = False
                continue

            max_val, min_val = -np.inf, np.inf
            for feature in example:  # find max and min features value
                if feature < min_val:
                    min_val = feature
                if feature > max_val:
                    max_val = feature

            for feature in range(len(example)):
                example[feature] = (example[feature] - min_val) / (max_val - min_val)
            min_max_values.append((min_val, max_val))

        return min_max_values

    def _test_minmax_normalize(self, test_examples: Examples):
        max_min_index = 0
        is_classifier = True
        for example in test_examples.transpose():
            if is_classifier:
                is_classifier = False
                continue

            for feature in range(len(example)):
                example[feature] = (example[feature] - self._min_max_values[max_min_index][0]) / \
                                   (self._min_max_values[max_min_index][1] - self._min_max_values[max_min_index][0])
            max_min_index += 1


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def main():
    print(KNN(TRAIN_PATH).classify(TEST_PATH))


if __name__ == "__main__":
    main()
