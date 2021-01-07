"""
ID3 Algorithm
"""
from utils import *
from math import log2

""""""""""""""""""""""""""""""""""""""""""" ID3 """""""""""""""""""""""""""""""""""""""""""


class ID3ContinuousFeatures(object):
    @staticmethod
    def learn_without_pruning(train_path: str, test_path: str) -> float:
        train_examples, train_features = get_full_examples_from_csv(train_path)
        return ID3ContinuousFeatures.get_accuracy(
            ID3ContinuousFeatures.get_classifier(train_examples, train_features),
            get_generator_examples_from_csv(test_path))

    @staticmethod
    def learn_with_pruning(train_path: str, test_path: str) -> float:
        M, train_examples, train_features = experiment(train_path, True)  # TODO: set diffrent if we dont want graph

        return ID3ContinuousFeatures.get_accuracy(
            ID3ContinuousFeatures.get_classifier(train_examples, train_features, M),
            get_generator_examples_from_csv(test_path))

    ######### Helper Functions for ID3 Algorithm #########
    @staticmethod
    def get_classifier(examples: Examples, features: Features, M: int = 1) -> Classifier:
        return ID3ContinuousFeatures._tdidt_algorithm(examples, features,
                                                      ID3ContinuousFeatures._majority_class(examples),
                                                      ID3ContinuousFeatures._max_ig, M)

    @staticmethod
    def get_accuracy(classifier: Classifier, examples: Examples) -> float:
        true_pos, true_neg = 0, 0
        test_examples_amount = 0
        for example in examples:
            example_result = ID3ContinuousFeatures._test_example(classifier, example)
            if example_result == 1 and example[0] == 1:
                true_pos += 1
            elif example_result == 0 and example[0] == 0:
                true_neg += 1
            test_examples_amount += 1

        return (true_pos + true_neg) / test_examples_amount

    @staticmethod
    def _tdidt_algorithm(examples: Examples, features: Features, default: int,
                         select_feature: Callable[[Examples], Tuple[int, Examples, Examples]],
                         M: int) -> Classifier:
        # Empty leaf
        if len(examples) == 0:
            return 0, [], default

        # Consistent node turns leaf
        majority_class = ID3ContinuousFeatures._majority_class(examples)
        if len(examples) <= M or features.size == 0 or ID3ContinuousFeatures._check_consistent_node(examples,
                                                                                                    majority_class):
            return 0, [], majority_class

        # main decision
        # dynamic_features = ID3ContinuousFeatures._continuous_features(features)
        chosen_feature, class_one, class_two = select_feature(examples)

        if class_one.size == 0 or class_two.size == 0:  # all the features are same- noise
            return 0, [], majority_class

        # create subtrees fits to the chosen_feature
        subtrees = [ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_one, chosen_feature, 1),
                                                           np.delete(features, chosen_feature - 1),
                                                           majority_class,
                                                           select_feature, M),

                    ID3ContinuousFeatures._tdidt_algorithm(np.delete(class_two, chosen_feature, 1),
                                                           np.delete(features, chosen_feature - 1),
                                                           majority_class,
                                                           select_feature, M)]

        return features[chosen_feature - 1], subtrees, majority_class

    @staticmethod
    def _max_ig(examples: Examples) -> Tuple[int, Examples, Examples]:
        father_entropy = ID3ContinuousFeatures._entropy(examples)

        max_class_one = []
        max_class_two = []
        max_ig = -np.inf
        argmax_ig = 0

        for feature in range(1, len(np.transpose(examples))):
            class_true, class_false = ID3ContinuousFeatures._divide_by_feature(examples, feature)
            son1_true_entropy = ID3ContinuousFeatures._entropy(class_true)
            son2_false_entropy = ID3ContinuousFeatures._entropy(class_false)
            ig = father_entropy - (son1_true_entropy * (len(class_true) / len(examples)) + son2_false_entropy * (
                    len(class_false) / len(examples)))

            if max_ig <= ig:
                max_ig = ig
                argmax_ig = feature
                max_class_one = class_true
                max_class_two = class_false

        return argmax_ig, max_class_one, max_class_two

    # @staticmethod
    # def _continuous_features(features: Features) -> Features:
    #     print(features)
    #     continuous_features = []
    #
    #     for feature in features.transpose():
    #         features_values = sorted(feature)
    #
    #         parts = []
    #         for part in range(len(features_values)-1):
    #             parts.append((features_values[part]+features_values[part+1])//2)
    #
    #         continuous_feature = []
    #         for part in parts:
    #             for old_feature in features_values:
    #                 continuous_feature.append(0 if old_feature <= part else 1)
    #
    #         continuous_features.append(continuous_feature)
    #
    #     return np.array(continuous_features).transpose()

    ######### Helper Functions in class #########
    @staticmethod
    def _divide_by_feature(examples: Examples, feature: int) -> Tuple[Examples, Examples]:
        class_true = []
        class_false = []
        example_num = 0
        for i in np.transpose(examples)[feature]:
            if i == 1:
                class_true.append(examples[example_num])
            elif i == 0:
                class_false.append(examples[example_num])
            example_num += 1

        return np.array(class_true), np.array(class_false)

    @staticmethod
    def _majority_class(examples: Examples) -> int:
        # assume examples != []
        num_true = np.count_nonzero((examples.transpose())[0])
        if num_true > len(examples) - num_true:
            return 1
        elif num_true < len(examples) - num_true:
            return 0
        else:
            return randint(0, 1)

    @staticmethod
    def _check_consistent_node(examples: Examples, c: int) -> bool:
        # assume examples != []
        for example in np.transpose(examples)[0]:
            if example != c:
                return False
        return True

    @staticmethod
    def _entropy(examples: Examples) -> float:
        if examples.size == 0:
            return 0
        num_true = np.count_nonzero((examples.transpose())[0])
        p_true = num_true / len(examples)
        p_false = 1 - p_true

        if p_true == 1.:
            p_false = 1
        elif p_true == 0.:
            p_true = 1

        return -(p_true * log2(p_true) + p_false * log2(p_false))

    @staticmethod
    def _test_example(classifier: Classifier, example: Examples) -> int:
        # assume classifier seems like: ( chosen_feature, [children: true, false], classification)
        if len(classifier[1]) == 0:  # if children == []: take classification
            return classifier[2]

        if example[classifier[0]]:
            return ID3ContinuousFeatures._test_example(classifier[1][0], example)
        return ID3ContinuousFeatures._test_example(classifier[1][1], example)


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def experiment(train_path: str = TRAIN_PATH, do_print_graph: bool = True) -> Tuple[int, Examples, Features]:
    """
        For using this function and print the graph, you may insert the path for train csv file and set 'do_print_graph' param
        for deciding to print the graph or not. In default, the function will print the graph for data in "./train.csv".

        @:param train_path(str): path for train data, default value: "./train.csv".
        @:param do_print_graph(bool): if true, the function will print the graph, otherwise the function will not. default value: True
        @:return the best M hyper-parameter, train examples and features (we don't want to read it again)
    """
    train_examples, train_features = get_full_examples_from_csv(train_path)
    folds = KFold(n_splits=N_SPLIT, shuffle=SHUFFLE, random_state=RANDOM_STATE)
    m_values = [i for i in range(2, 7)]
    m_accuracy = []

    for m_value in m_values:
        accuracy = 0
        for train_fold, test_fold in folds.split(train_examples):
            classifier = ID3ContinuousFeatures.get_classifier(np.take(train_examples, train_fold, 0), train_features, m_value)
            accuracy += ID3ContinuousFeatures.get_accuracy(classifier, np.take(train_examples, test_fold, 0))
        m_accuracy.append(accuracy / N_SPLIT)

    if do_print_graph:
        print_graph(m_values, m_accuracy)

    assert len(m_values) == N_SPLIT  # TODO: Delete

    return m_values[int(np.argmax(m_accuracy))], train_examples, train_features


def pruning_test():
    create_test(100,1000, "train")
    create_test(100,1000, "test")
    print(ID3ContinuousFeatures.learn_without_pruning("./test_csv/train.csv", "./test_csv/test.csv"))
    print(ID3ContinuousFeatures.learn_with_pruning("./test_csv/train.csv", "./test_csv/test.csv"))


def main():
    pruning_test()


if __name__ == "__main__":
    main()
