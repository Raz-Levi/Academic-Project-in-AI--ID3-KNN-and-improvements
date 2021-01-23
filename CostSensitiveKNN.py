"""
CostSensitiveKNN Algorithm
"""
from KNN import KNN
from utils import Examples, TRAIN_PATH, TEST_PATH, CommitteeWrapper, insort, euclidean_distance
from ID3 import ID3ContinuousFeatures

BEST_LOSS_K = 1
BEST_BOUND = 0.056

""""""""""""""""""""""""""""""""""""""""""" CostSensitiveKNN """""""""""""""""""""""""""""""""""""""""""


class CostSensitiveKNN(KNN):
    def __init__(self, train_path: str, bound: float = BEST_BOUND, k: int = BEST_LOSS_K):
        super().__init__(train_path, k)
        self._bound = bound
        self._id3_classifier = None

    def classify(self, test_path: str) -> float:
        return self.classify_and_get_loss(test_path)

    def classify_one(self, test_example: Examples) -> int:
        committee = []
        for train_example in self._train_examples:
            insort(committee, CommitteeWrapper(train_example[0], euclidean_distance(test_example, train_example)))

        votes_num, vote_for, vote_against, under_bound = 0, 0, 0, (0, 0)
        for vote in committee:
            if vote.distance < self._bound:
                under_bound = (vote_for, vote_against)
            if votes_num >= self._k:
                break
            if vote == 1:
                vote_for += 1
            else:
                vote_against += 1
            votes_num += 1

        if vote_for >= vote_against:
            return 1

        # KNN wants to classify as Negative to disease (healthy), let's get a second opinion
        if self._id3_classifier is None:
            self._id3_classifier = ID3ContinuousFeatures(self._train_examples)

        if self._id3_classifier.classify_one(test_example):
            return 1

        # Both KNN and ID3 want to classify as Negative to disease (healthy), let's get a third final opinion
        return under_bound[0] > under_bound[1]

    ######### Private Functions for CostSensitiveKNN Algorithm #########
    def _get_accuracy(self, test_examples: Examples) -> float:
        return self._get_loss(test_examples)


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def main():
    print(CostSensitiveKNN(TRAIN_PATH).classify(TEST_PATH))


if __name__ == "__main__":
    main()
