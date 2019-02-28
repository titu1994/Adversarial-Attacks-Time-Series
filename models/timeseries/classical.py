import classical.classification.DTW as _DTW
from utils.generic_utils import BaseClassicalModel


class DTW(BaseClassicalModel):

    def __init__(self, *args, neighbours=1, name=None, **kwargs):
        super(DTW, self).__init__(name=name, **kwargs)

        self.neighbours = neighbours
        self.model = _DTW.KnnDTW(neighbours)

    def fit(self, X, Y, training=True, **kwargs):
        self.model.fit(X, Y)

    def predict(self, X, training=False, **kwargs):
        mode_label, mode_proba = self.model.predict(X)
        return mode_label


class DTWProbabilistic(BaseClassicalModel):

    def __init__(self, *args, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        super(DTWProbabilistic, self).__init__(name=name, **kwargs)

        self.neighbours = 1
        self.model = _DTW.KnnDTW(self.neighbours)

    def fit(self, X, Y, training=True, **kwargs):
        self.model.fit(X, Y)

    def predict(self, X, training=False, **kwargs):
        probas, labels = self.model.predict_proba(X)
        return probas


