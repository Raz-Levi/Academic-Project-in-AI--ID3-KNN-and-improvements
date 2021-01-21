"""
ID3 Algorithm
"""
from LearningAlgorithm import LearningAlgorithm
from utils import *
from math import log2

BEST_M = 1

""""""""""""""""""""""""""""""""""""""""""" ID3 """""""""""""""""""""""""""""""""""""""""""


class ID3ContinuousFeatures(LearningAlgorithm):
    def __init__(self, train_examples, m: int = BEST_M):
        if type(train_examples) == str:
            super().__init__(train_examples)
        else:
            self._train_examples = train_examples
        self._m = m
        self._classifier = self.get_classifier()

    def classify(self, test_examples) -> float:
        if type(test_examples) == str:
            return self._get_accuracy(get_generator_examples_from_csv(test_examples))
        return self._get_accuracy(test_examples)

    def get_classifier(self) -> Classifier:
        return self._tdidt_algorithm(self._train_examples, self._majority_class(self._train_examples)[0], self._max_ig_continuous_features)

    ######### Helper Functions for ID3 Algorithm #########
    def _classify_one(self, test_example: Examples) -> int:
        return self._classify_one_recursive(self._classifier, test_example)

    def _tdidt_algorithm(self, train_examples: Examples, default: int,
                         select_feature: Callable[[Examples], Tuple[Tuple[int, float], Examples, Examples]]) -> Classifier:

        # Empty leaf
        if len(train_examples) == 0 or len(train_examples) < self._m:
            return (0, 0), [], default

        # Consistent node turns leaf
        majority_class, num_of_majority_class = ID3ContinuousFeatures._majority_class(train_examples)
        if len(train_examples[0]) == 1 or num_of_majority_class == len(train_examples):
            return (0, 0), [], majority_class

        # main decision
        chosen_feature, class_one, class_two = select_feature(train_examples)
        if class_one.size == 0 or class_two.size == 0:  # all the features are same- noise
            return (0, 0), [], majority_class

        # create subtrees fits to the chosen_feature
        subtrees = [self._tdidt_algorithm(class_one, majority_class, select_feature),
                    self._tdidt_algorithm(class_two, majority_class, select_feature)]

        return chosen_feature, subtrees, majority_class

    ######### Static Functions for ID3 Algorithm #########
    @staticmethod
    def _classify_one_recursive(classifier: Classifier, test_example: Examples) -> int:
        # assume classifier seems like: ( (chosen_feature, part), [children: true, false], classification)
        if len(classifier[1]) == 0:  # if children == []: take classification
            return classifier[2]

        if test_example[classifier[0][0]] > classifier[0][1]:
            return ID3ContinuousFeatures._classify_one_recursive(classifier[1][0], test_example)
        return ID3ContinuousFeatures._classify_one_recursive(classifier[1][1], test_example)

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

                if max_ig <= ig:
                    max_ig = ig
                    argmax_ig = feature
                    max_class_one = class_true
                    max_class_two = class_false
                    max_part = binary_feature[1]

        return (argmax_ig, max_part), max_class_one, max_class_two

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


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def experiment(train_path: str = TEST_PATH, do_print_graph: bool = False) -> int:
    """
        For using this function and print the graph, you may call this method with any path to train examples you wish
        (default is 'train.csv') and set 'do_print_graph' param to True.
        In default, the function will train on 'train.csv' and will not print the graph.

        @:param train_path(str): path for train data. default value: 'train.csv'.
        @:param do_print_graph(bool): if true, the function will print the graph, otherwise the function will not. default value: False.
        @:return the best M hyper-parameter.
    """
    train_examples = get_full_examples_from_csv(train_path)
    folds = KFold(n_splits=N_SPLIT, shuffle=SHUFFLE, random_state=RANDOM_STATE)
    m_values = M_VALUES
    m_accuracy = []

    for m_value in m_values:
        accuracy = 0
        for train_fold, test_fold in folds.split(train_examples):
            accuracy += ID3ContinuousFeatures(np.take(train_examples, train_fold, 0), m_value).classify(np.take(train_examples, test_fold, 0))
        m_accuracy.append(accuracy / N_SPLIT)

    if do_print_graph:
        print_graph(m_values, m_accuracy, 'M')

    return m_values[int(np.argmax(m_accuracy))]


def main():
    print(ID3ContinuousFeatures(TRAIN_PATH).classify(TEST_PATH))


if __name__ == "__main__":
    main()
