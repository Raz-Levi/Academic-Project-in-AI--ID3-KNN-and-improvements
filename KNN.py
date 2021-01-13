"""
KNN Algorithm
"""
from LearningAlgorithm import LearningAlgorithm
from utils import get_full_examples_from_csv, Examples, euclidean_distance, TRAIN_PATH, TEST_PATH, np, insort

""""""""""""""""""""""""""""""""""""""""""" KNN """""""""""""""""""""""""""""""""""""""""""


class KNN(LearningAlgorithm):
    def __init__(self, train_path: str, k: int = 1):
        super().__init__(train_path)
        self._min_max_values = self._train_minmax_normalize()
        self._k = k

    def classify(self, test_path: str) -> float:
        test_examples = get_full_examples_from_csv(test_path)
        self._test_minmax_normalize(test_examples)
        # assert test_examples == self._train_examples # TODO: Delete
        return self._get_accuracy(test_examples)

    def classify_and_get_loss(self, test_path) -> float:
        test_examples = get_full_examples_from_csv(test_path)
        self._test_minmax_normalize(test_examples)
        return self._get_loss(test_examples)

    def _get_accuracy(self, test_examples: Examples) -> float:
        classify_correct = 0
        for example in test_examples:
            if self._classify_one(example) == example[0]:
                classify_correct += 1

        return classify_correct / len(test_examples)

    def _get_loss(self, test_examples: Examples) -> float:
        fp, fn = 0, 0
        for example in test_examples:
            example_result = self._classify_one(example)
            if example_result == 1 and example[0] == 0:
                fp += 1
            elif example_result == 0 and example[0] == 1:
                fn += 1

        return (0.1 * fp + fn) / len(test_examples)

    ######### Helper Functions for KNN Algorithm #########
    def _classify_one(self, example: Examples) -> int:

        class DistanceWrapper(object):
            def __init__(self, classification: int, distance: float):
                self.classification = classification
                self.distance = distance

            def __lt__(self, other):
                return self.distance < other.distance

            def __eq__(self, other: int):
                return self.classification == other

        distances = []
        for train_example in self._train_examples:
            insort(distances, DistanceWrapper(train_example[0], euclidean_distance(example, train_example)))

        votes_num, vote_true, vote_false = 0, 0, 0
        for vote in distances:
            if votes_num >= self._k:
                break
            if vote == 1:
                vote_true += 1
            else:
                vote_false += 1
            votes_num += 1
        return 1 if vote_true >= vote_false else 0

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

    def _test_minmax_normalize(self, test_examples: Examples):  # TODO: check if succeeded normalize by ref
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


# def experiment(train_examples: Examples, do_print_graph: bool = False) -> int:
#     """
#         For using this function and print the graph, you may use 'KNN.classify' function and set 'do_print_graph' param
#         to True. In default, the function will not print the graph.
#
#         @:param train_examples(np.array): the train examples.
#         @:param do_print_graph(bool): if true, the function will print the graph, otherwise the function will not.
#         @:return the best K.
#     """
#     folds = KFold(n_splits=N_SPLIT, shuffle=SHUFFLE, random_state=RANDOM_STATE)
#     k_values = [i for i in range(1, NUM_FOR_CHOOSE+1)]  # assume len(train_examples) >= 5, because we can't KFold with n_split = 5
#     k_accuracy = []
#
#     for k_value in k_values:
#         accuracy = 0
#         for train_fold, test_fold in folds.split(train_examples):
#             accuracy += KNN.get_accuracy(np.take(train_examples, train_fold, 0), np.take(train_examples, test_fold, 0), k_value)
#         k_accuracy.append(accuracy / N_SPLIT)
#
#     if do_print_graph:
#         print_graph(k_values, k_accuracy, 'K')
#
#     assert len(k_values) == N_SPLIT  # TODO: Delete
#
#     return k_values[int(np.argmax(k_accuracy))]


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def main():
    print(KNN(TRAIN_PATH).classify(TEST_PATH))


if __name__ == "__main__":
    main()
