"""
KNN Algorithm
"""
from utils import *
from bisect import insort

""""""""""""""""""""""""""""""""""""""""""" KNN """""""""""""""""""""""""""""""""""""""""""


class KNN(object):
    def __init__(self, train_path: str, print_graph: bool = True):  # TODO: what about choosing features?
        non_normalized_examples, self.train_features = get_generator_examples_from_csv(train_path)
        self.train_examples = KNN._minmax_normalize(non_normalized_examples)
        self.k = KNN.experiment(print_graph, train_path)

    def classify(self, example: Examples) -> int:

        class DistanceWrapper(object):
            def __init__(self, classification: int, distance: float):
                self.classification = classification
                self.distance = distance

            def __lt__(self, other):
                return self.distance < other.distance

            def __eq__(self, other: int):
                return self.classification == other

        distances = []
        for train_example in self.train_examples:
            insort(distances, DistanceWrapper(train_example[0], KNN._euclidean_distance(example, train_example)))

        vote_true = np.count_nonzero(distances[len(distances)-(self.k - 1):] == 1)
        return 1 if vote_true >= self.k - vote_true else 0  # TODO: what about tie?

    def get_accuracy(self, test_path: str) -> float:
        test_examples = get_generator_examples_from_csv(test_path)
        true_pos, true_neg = 0, 0
        test_examples_amount = 0
        for example in test_examples:
            example_result = self.classify(example)
            if example_result == 1 and example[0] == 1:
                true_pos += 1
            elif example_result == 0 and example[0] == 0:
                true_neg += 1
            test_examples_amount += 1

        return (true_pos + true_neg) / test_examples_amount

    ######### Helper Functions for KNN Algorithm #########
    @staticmethod
    def _minmax_normalize(examples: Examples) -> Examples:
        normalized_examples = []
        for example in examples:

            features = examples[1:]
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
    def _euclidean_distance(example_one: Examples, example_two: Examples) -> float:
        # assume len(example_one) == len(example_two)
        euclidean_distance = 0
        is_feature = False
        for feature_one, feature_two in zip(example_one, example_two):  # the first cell in example is not a feature
            if not is_feature:
                is_feature = True
                continue
            euclidean_distance += (feature_one - feature_two) ** 2

        return euclidean_distance ** 0.5

    @staticmethod
    def experiment(print_graph: bool, train_path: str = TRAIN_PATH) -> Tuple[int, Examples, Features]: # TODO: Finish it!
        """
            For using this function and print the graph, you may insert the path for train csv file and set 'print_graph' param
            for deciding to print the graph or not. In default, the function will print the graph for data in "./train.csv".

            @:param train_path(str): path for train data, default value: "./train.csv".
            @:param print_graph(bool): if true, the function will print the graph, otherwise the function will not. default value: True
            @:return the best M hyper-parameter, train examples and features (we don't want to read it again)
        """
        train_examples, train_features = get_full_examples_from_csv(train_path)
        folds = KFold(n_splits=N_SPLIT, shuffle=SHUFFLE, random_state=RANDOM_STATE)
        m_values = [i for i in range(3, 12, 2)]
        m_accuracy = []

        for m_value in m_values:  # TODO: which M values should I choose? Random?
            accuracy = 0
            for train_fold, test_fold in folds.split(train_examples):
                knn = KNN()
                accuracy += KNN.get_accuracy()
            m_accuracy.append(accuracy / N_SPLIT)

        if print_graph:
            plt.plot(m_values, m_accuracy)
            plt.ylabel('Average accuracy')
            plt.xlabel('M values')
            plt.show()

            print(m_values[int(np.argmax(m_accuracy))])  # TODO: Delete
        assert len(m_values) == N_SPLIT  # TODO: Delete

        return m_values[int(np.argmax(m_accuracy))], train_examples, train_features


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def main():
    pass


if __name__ == "__main__":
    main()
