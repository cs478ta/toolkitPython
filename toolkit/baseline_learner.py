from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix


class BaselineLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    def __init__(self):
        # Your models should initialize weights and other model frameworks here
        #self. weights = np.random.random((size))
        self.average_label = []

    def train(self, features, labels):
        """
        This function should loop through the data and create/update a ML model until some stopping criteria is reached
        :type features: Matrix
        :type labels: Matrix
        """
        self.labels = []
        for i in range(labels.cols):
            if labels.value_count(i) == 0:
                self.average_label += [labels.column_mean(i)]          # continuous
            else:
                self.average_label += [labels.most_common_value(i)]    # nominal

    def predict(self, features):
        """
        This function runs 1 instance through the model and returns the model's predicted label
        :type features: [float]
        """
        return self.average_label


