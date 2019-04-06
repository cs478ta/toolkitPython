from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner
from toolkit.baseline_learner import BaselineLearner
from toolkit.arff import Arff
from toolkit import utils
import os
import numpy as np

class TestBaselineLearner(TestCase):
    def setUp(self):
        path = os.path.join(utils.get_root(), "test/datasets/cm1_req.arff")
        data = Arff(arff=path)

        self.features = data.get_features()
        self.labels = data.get_labels()
        self.learner = BaselineLearner()

    def test_train(self):
        self.learner.train(self.features, self.labels)
        self.assertAlmostEqual(self.learner.average_label[0], 0.0, places=4)

    def test_predict(self):
        self.learner.train(self.features, self.labels)
        label = self.learner.predict_all(self.features[0])[0] # just check first one
        np.testing.assert_almost_equal(self.learner.average_label, label)

if __name__=="__main__":
    suite = TestLoader().loadTestsFromTestCase(TestBaselineLearner)
    TextTestRunner(verbosity=2).run(suite)
