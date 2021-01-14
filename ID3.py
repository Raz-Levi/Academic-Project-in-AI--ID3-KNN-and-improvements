"""
ID3 Algorithm
"""
from utils import *
from math import log2

BEST_M = 1

""""""""""""""""""""""""""""""""""""""""""" ID3 """""""""""""""""""""""""""""""""""""""""""


class ID3ContinuousFeatures(object):
    @staticmethod
    def classify_without_pruning(train_path: str, test_path: str) -> float:
        train_examples = get_full_examples_from_csv(train_path)
        return ID3ContinuousFeatures.get_accuracy(
            ID3ContinuousFeatures.get_classifier(train_examples),
            get_generator_examples_from_csv(test_path))

    @staticmethod
    def classify_with_pruning(train_path: str, test_path: str, do_print_graph: bool = False) -> float:
        M, train_examples = ID3ContinuousFeatures.experiment(train_path, do_print_graph)

        return ID3ContinuousFeatures.get_accuracy(
            ID3ContinuousFeatures.get_classifier(train_examples, M),
            get_generator_examples_from_csv(test_path))

    ######### Helper Functions for ID3 Algorithm #########
    @staticmethod
    def get_classifier(examples: Examples, M: int = 1) -> Classifier:
        return ID3ContinuousFeatures._tdidt_algorithm(examples,
                                                      ID3ContinuousFeatures._majority_class(examples)[0],
                                                      ID3ContinuousFeatures._max_ig_continuous_features, M)

    @staticmethod
    def get_accuracy(classifier: Classifier, examples: Examples) -> float:
        classify_correct, test_examples_amount = 0, 0
        for example in examples:
            example_result = ID3ContinuousFeatures.classify_one(classifier, example)
            if example_result == example[0]:
                classify_correct += 1
            test_examples_amount += 1

        return classify_correct / test_examples_amount

    @staticmethod
    def classify_one(classifier: Classifier, example: Examples) -> int:
        # assume classifier seems like: ( (chosen_feature, part), [children: true, false], classification)
        if len(classifier[1]) == 0:  # if children == []: take classification
            return classifier[2]

        if example[classifier[0][0]] > classifier[0][1]:
            return ID3ContinuousFeatures.classify_one(classifier[1][0], example)
        return ID3ContinuousFeatures.classify_one(classifier[1][1], example)

    @staticmethod
    def _tdidt_algorithm(examples: Examples, default: int,
                         select_feature: Callable[[Examples], Tuple[Tuple[int, float], Examples, Examples]],
                         M: int) -> Classifier:
        # Empty leaf
        if len(examples) == 0 or len(examples) <= M:  # TODO: should be <=
            return (0, 0), [], default

        # Consistent node turns leaf
        majority_class, check_consistent_node = ID3ContinuousFeatures._majority_class(examples)
        if len(examples[0]) == 1 or check_consistent_node == len(examples):
            return (0, 0), [], majority_class

        # main decision
        chosen_feature, class_one, class_two = select_feature(examples)
        if class_one.size == 0 or class_two.size == 0:  # all the features are same- noise
            return (0, 0), [], majority_class

        # create subtrees fits to the chosen_feature
        subtrees = [ID3ContinuousFeatures._tdidt_algorithm(class_one, majority_class, select_feature, M),
                    ID3ContinuousFeatures._tdidt_algorithm(class_two, majority_class, select_feature, M)]

        return chosen_feature, subtrees, majority_class

    @staticmethod
    def experiment(train_path: str, do_print_graph: bool) -> Tuple[int, Examples]:
        """
            For using this function and print the graph, you may use 'ID3ContinuousFeatures.classify_with_pruning' function and set 'do_print_graph' param
            to True. In default, the function will not print the graph.

            @:param train_path(str): path for train data.
            @:param do_print_graph(bool): if true, the function will print the graph, otherwise the function will not. default value: False.
            @:return the best M hyper-parameter, train examples and features (we don't want to read it again).
        """
        train_examples = get_full_examples_from_csv(train_path)
        folds = KFold(n_splits=N_SPLIT, shuffle=SHUFFLE, random_state=RANDOM_STATE)
        m_values = [1, 2, 3, 5, 8, 16] # TODO: change to [i for i in range(4, NUM_FOR_CHOOSE + 4)]
        m_accuracy = []

        for m_value in m_values:
            accuracy = 0
            for train_fold, test_fold in folds.split(train_examples):
                classifier = ID3ContinuousFeatures.get_classifier(np.take(train_examples, train_fold, 0), m_value)
                accuracy += ID3ContinuousFeatures.get_accuracy(classifier, np.take(train_examples, test_fold, 0))
            m_accuracy.append(accuracy / N_SPLIT)

        if do_print_graph:
            print_graph(m_values, m_accuracy, 'M')

        print([(i,j) for i, j in zip(m_values, m_accuracy)])  # TODO: Delete
        print(m_values[int(np.argmax(m_accuracy))])  # TODO: Delete
        # assert len(m_values) == N_SPLIT  # TODO: Delete
        exit()

        return m_values[int(np.argmax(m_accuracy))], train_examples

    @staticmethod
    def _max_ig_continuous_features(examples: Examples) -> Tuple[Tuple[int, float], Examples, Examples]:
        father_entropy = ID3ContinuousFeatures._entropy(examples)

        max_class_one, max_class_two = [], []
        max_ig = -np.inf
        argmax_ig, max_part = 0, 0
        features = np.transpose(examples)

        for feature in range(1, len(features)):
            binary_features = ID3ContinuousFeatures._dynamic_partition(features[feature])
            for binary_feature in binary_features:
                class_true, class_false = ID3ContinuousFeatures._divide_by_feature(examples, binary_feature[0])
                son1_true_entropy = ID3ContinuousFeatures._entropy(class_true)
                son2_false_entropy = ID3ContinuousFeatures._entropy(class_false)
                ig = father_entropy - (son1_true_entropy * (len(class_true) / len(examples)) + son2_false_entropy * (
                        len(class_false) / len(examples)))

                if max_ig <= ig:  # TODO: With = or not?
                    max_ig = ig
                    argmax_ig = feature
                    max_class_one = class_true
                    max_class_two = class_false
                    max_part = binary_feature[1]

        return (argmax_ig, max_part), max_class_one, max_class_two


    ######### Helper Functions in class #########
    @staticmethod
    def _dynamic_partition(feature: Features) -> Examples:
        binary_features = []
        features_values = sorted(feature)

        parts = []
        for part in range(len(features_values) - 1):
            parts.append((features_values[part] + features_values[part + 1]) / 2)

        for part in parts:
            binary_feature = []
            for f in feature:
                binary_feature.append(0 if f < part else 1)
            binary_features.append((binary_feature, part))

        return np.array(binary_features)

    @staticmethod
    def _entropy(examples: Examples) -> float:
        if examples.size == 0:
            return 0
        num_true = np.count_nonzero((np.transpose(examples)[0]))
        p_true = num_true / len(examples)
        p_false = 1 - p_true

        if p_true == 1.:
            p_false = 1
        elif p_true == 0.:
            p_true = 1

        return -(p_true * log2(p_true) + p_false * log2(p_false))

    @staticmethod
    def _divide_by_feature(examples: Examples, feature: Features) -> Tuple[Examples, Examples]:
        class_true, class_false = [], []
        example_num = 0
        for i in feature:
            if i == 1:
                class_true.append(examples[example_num])
            elif i == 0:
                class_false.append(examples[example_num])
            example_num += 1

        return np.array(class_true), np.array(class_false)

    @staticmethod
    def _majority_class(examples: Examples) -> Tuple[int, int]:
        # assume examples != []
        num_true = np.count_nonzero((examples.transpose())[0])
        if num_true > len(examples) - num_true:
            return 1, num_true
        elif num_true < len(examples) - num_true:
            return 0, len(examples) - num_true
        else:
            return randint(0, 1), 0

    # @staticmethod
    # def _check_consistent_node(examples: Examples, c: int) -> bool:
    #     # assume examples != []
    #     for example in np.transpose(examples)[0]:
    #         if example != c:
    #             return False
    #     return True


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def main():
    print(ID3ContinuousFeatures.classify_without_pruning(TRAIN_PATH, TEST_PATH))


if __name__ == "__main__":
    main()

