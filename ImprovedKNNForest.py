"""
ImprovedKNNForest Algorithm
"""
from KNNForest import KNNForest, BEST_N, BEST_P, BEST_K
from utils import Examples, TRAIN_PATH, TEST_PATH, CommitteeWrapper, euclidean_distance, insort

BEST_AREA_LENGTH = 1
BEST_VOTE_WEIGHT = 2

""""""""""""""""""""""""""""""""""""""""""" ImprovedKNNForest """""""""""""""""""""""""""""""""""""""""""


class ImprovedKNNForest(KNNForest):
    def __init__(self, train_path: str, optimal_distance: float = BEST_AREA_LENGTH, vote_weight: int = BEST_VOTE_WEIGHT,
                 num_trees: int = BEST_N, p: float = BEST_P, num_chosen_trees: int = BEST_K):
        super().__init__(train_path, num_trees, p, num_chosen_trees)
        self._area_length, self._vote_weight = optimal_distance, vote_weight

    def classify_one(self, test_example: Examples) -> int:
        committee = []
        for tree in self._forest:
            insort(committee, CommitteeWrapper(tree.classifier, euclidean_distance(tree.centroid, test_example[1:])))

        votes_num, vote_for, vote_against, distance_range, vote_weight = 0, 0, 0, 1, 1
        for classifier in committee:
            if votes_num >= self._num_chosen_trees:
                break
            if classifier.distance > distance_range * self._area_length:
                vote_weight /= self._vote_weight
                distance_range += 1
            if classifier.classification_or_classifier.classify_one(test_example) == 1:
                vote_for += vote_weight
            else:
                vote_against += vote_weight
            votes_num += 1
        return 1 if vote_for >= vote_against else 0


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def main():
    print(ImprovedKNNForest(TRAIN_PATH).classify(TEST_PATH))


if __name__ == '__main__':
    # main()
    experiment_loop()
