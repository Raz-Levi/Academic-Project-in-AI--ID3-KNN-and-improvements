"""
KNN Algorithm
"""
from utils import *

""""""""""""""""""""""""""""""""""""""""""" KNN """""""""""""""""""""""""""""""""""""""""""


class KNN(object):
    @staticmethod
    def classify(train_path: str, test_path, do_print_graph: bool = False) -> float:
        train_examples = KNN._minmax_normalize(get_full_examples_from_csv(train_path, get_features=False)[0])
        test_examples = KNN._minmax_normalize(get_generator_examples_from_csv(test_path))
        return KNN.get_accuracy(train_examples, test_examples, k=KNN.experiment(train_examples, do_print_graph))

    @staticmethod
    def classify_and_get_loss(train_path: str, test_path, do_print_graph: bool = False) -> float:
        train_examples = KNN._minmax_normalize(get_full_examples_from_csv(train_path, get_features=False)[0])
        test_examples = KNN._minmax_normalize(get_generator_examples_from_csv(test_path))
        return KNN.get_loss(train_examples, test_examples, k=KNN.experiment(train_examples, do_print_graph))

    @staticmethod
    def get_accuracy(train_examples: Examples, test_examples: Examples, k: int) -> float:
        true_pos, true_neg = 0, 0
        test_examples_amount = 0
        for example in test_examples:
            example_result = KNN._classify_one(train_examples, example, k)
            if example_result == 1 and example[0] == 1:
                true_pos += 1
            elif example_result == 0 and example[0] == 0:
                true_neg += 1
            test_examples_amount += 1

        return (true_pos + true_neg) / test_examples_amount

    @staticmethod
    def get_loss(train_examples: Examples, test_examples: Examples, k: int) -> float:
        fp, fn = 0, 0
        test_examples_amount = 0
        for example in test_examples:
            example_result = KNN._classify_one(train_examples, example, k)
            if example_result == 1 and example[0] == 0:
                fp += 1
            elif example_result == 0 and example[0] == 1:
                fn += 1
            test_examples_amount += 1

        return (0.1*fp + fn) / test_examples_amount

    ######### Helper Functions for KNN Algorithm #########
    @staticmethod
    def _classify_one(train_examples: Examples, example: Examples, k: int) -> int:

        class DistanceWrapper(object):
            def __init__(self, classification: int, distance: float):
                self.classification = classification
                self.distance = distance

            def __lt__(self, other):
                return self.distance < other.distance

            def __eq__(self, other: int):
                return self.classification == other

        distances = []
        for train_example in train_examples:
            insort(distances, DistanceWrapper(train_example[0], euclidean_distance(example, train_example)))

        votes_num = 0
        vote_true = 0
        for vote in distances:
            if votes_num >= k:
                break
            if vote == 1:
                vote_true += 1
            votes_num += 1
        return 1 if vote_true >= min(k, len(distances)) - vote_true else 0

    @staticmethod
    def _minmax_normalize(examples: Examples) -> Examples:
        normalized_examples = []
        for example in examples:

            features = np.delete(example, 0)
            max_val, min_val = -np.inf, np.inf
            for feature in features:  # find max and min features value
                if feature < min_val:
                    min_val = feature
                elif feature > max_val:
                    max_val = feature

            normalized_example = [example[0]]
            for feature in features:
                normalized_example.append((feature - min_val) / (max_val - min_val))

            normalized_examples.append(normalized_example)

        return np.array(normalized_examples)

    @staticmethod
    def experiment(train_examples: Examples, do_print_graph: bool) -> int:
        """
            For using this function and print the graph, you may use 'KNN.classify' function and set 'do_print_graph' param
            to True. In default, the function will not print the graph.

            @:param train_examples(np.array): the train examples.
            @:param do_print_graph(bool): if true, the function will print the graph, otherwise the function will not.
            @:return the best K.
        """
        folds = KFold(n_splits=N_SPLIT, shuffle=SHUFFLE, random_state=RANDOM_STATE)
        k_values = [i for i in range(1, NUM_FOR_CHOOSE+1)]  # assume len(train_examples) >= 5, because we can't KFold with n_split = 5
        k_accuracy = []

        for k_value in k_values:
            accuracy = 0
            for train_fold, test_fold in folds.split(train_examples):
                accuracy += KNN.get_accuracy(np.take(train_examples, train_fold, 0), np.take(train_examples, test_fold, 0), k_value)
            k_accuracy.append(accuracy / N_SPLIT)

        if do_print_graph:
            print_graph(k_values, k_accuracy, 'K')

        assert len(k_values) == N_SPLIT  # TODO: Delete

        return k_values[int(np.argmax(k_accuracy))]


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


# TODO: Delete!
def learn_k(train_path: str, test_path) -> float:
    train_examples = KNN._minmax_normalize(get_full_examples_from_csv(train_path, get_features=False)[0])
    test_examples = KNN._minmax_normalize(get_generator_examples_from_csv(test_path))
    return KNN.get_accuracy(train_examples, test_examples, 1)


def main():
    print(KNN.classify(TRAIN_PATH, TEST_PATH))


if __name__ == "__main__":
    main()
