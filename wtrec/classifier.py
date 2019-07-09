from sklearn.ensemble import GradientBoostingClassifier


RANDOM_STATE = 2 ** 16 - 1


class BaselineClassifier(object):
    """
    GBDT classifier as specified in baseline paper
    
    Wrapper around scikit-learn classifier, implements same interface
    """
    def __init__(self):
        super().__init__()
        self._clf = GradientBoostingClassifier(
            min_samples_leaf=2,
            n_estimators=100,
            random_state=RANDOM_STATE,
        )

    def fit(self, X, Y):
        self._clf.fit(X, Y)
        return self._clf

    def predict(self, X):
        return self._clf.predict(X)
