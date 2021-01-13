"""
CostSensitiveKNN Algorithm
"""
from KNN import KNN
from utils import Examples, TRAIN_PATH, TEST_PATH, insort, euclidean_distance
from ID3 import ID3ContinuousFeatures

BEST_BOUND = 0.056

""""""""""""""""""""""""""""""""""""""""""" CostSensitiveKNN """""""""""""""""""""""""""""""""""""""""""


class CostSensitiveKNN(KNN):
    def __init__(self, train_path: str, bound: float = BEST_BOUND):
        super().__init__(train_path)
        self._bound = bound
        self._id3_classifier = None

    def classify(self, test_path: str) -> float:
        return self.classify_and_get_loss(test_path)

    def _classify_one(self, test_example: Examples) -> int:

        class DistanceWrapper(object):
            def __init__(self, classification: int, distance: float):
                self.classification = classification
                self.distance = distance

            def __lt__(self, other):
                return self.distance < other.distance

            def __eq__(self, other: int):
                return self.classification == other

        committee = []
        for train_example in self._train_examples:
            insort(committee, DistanceWrapper(train_example[0], euclidean_distance(test_example, train_example)))

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
            self._id3_classifier = ID3ContinuousFeatures.get_classifier(self._train_examples)

        if ID3ContinuousFeatures.classify_one(self._id3_classifier, test_example):
            return 1

        # Both KNN and ID3 want to classify as Negative to disease (healthy), let's get a third opinion
        return under_bound[0] > under_bound[1]


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def main():
    print(CostSensitiveKNN(TRAIN_PATH).classify(TEST_PATH))


if __name__ == "__main__":
    main()
