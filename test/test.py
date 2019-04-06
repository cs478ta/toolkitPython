from __future__ import (absolute_import, division, print_function, unicode_literals)
from unittest import TestCase,TestLoader,TextTestRunner
import numpy as np
import os
from toolkit import baseline_learner, utils, manager, arff

class TestManager(TestCase):
    infinity = float("infinity")

    def setUp(self):
        self.arff_path = "../test/datasets/creditapproval.arff"
        self.my_learner = baseline_learner.BaselineLearner

    def test_train(self):
        my_learner = baseline_learner.BaselineLearner
        session = manager.ToolkitSession(arff=self.arff_path, learner=my_learner, eval_method=None,
                                         eval_parameter=None, print_confusion_matrix=False, normalize=False,
                                         random_seed=None, label_count=1)
        session.train()
        self.assertAlmostEqual(session.training_accuracy[0],0.55507246)

    def test_train_test_manual(self):
        my_learner = baseline_learner.BaselineLearner
        session = manager.ToolkitSession(arff=self.arff_path, learner=my_learner, eval_method=None,
                                         eval_parameter=None, print_confusion_matrix=True, normalize=False,
                                         random_seed=None, label_count=1)

        train_features, train_labels, test_features, test_labels = session.training_test_split(.7) # 70% training
        session.train(train_features, train_labels)
        session.test(test_features, test_labels)

    def test_sse(self):
        my_learner = baseline_learner.BaselineLearner
        session = manager.ToolkitSession(arff=self.arff_path, learner=my_learner, label_count=1)
        train_features, train_labels, test_features, test_labels = session.training_test_split(.7) # 70% training
        session.train(train_features, train_labels)
        session.learner.measure_accuracy(test_features, test_labels, eval_method="sse")


    def test_handle_numpy_array(self):
        """ If arrays are passed to measure accuracy instead of arff objects
        """
        my_learner = baseline_learner.BaselineLearner
        session = manager.ToolkitSession(arff=self.arff_path, learner=my_learner, label_count=1)
        train_features, train_labels, test_features, test_labels = session.training_test_split(.7) # 70% training
        session.train(train_features, train_labels)
        session.learner.measure_accuracy(test_features.data, test_labels.data) # will test accuracy, assumes no nominal variables
        session.learner.measure_accuracy(test_features.data, test_labels.data, eval_method="sse")

        # Test naive confusion matrix
        session.learner.measure_accuracy(test_features.data, test_labels.data)

    def test_train_test(self):
        my_learner = baseline_learner.BaselineLearner
        session = manager.ToolkitSession(arff=self.arff_path, learner=my_learner, eval_method="training",
                                         eval_parameter=None, print_confusion_matrix=False, normalize=False,
                                         random_seed=None, label_count=1)
        self.assertAlmostEqual(session.training_accuracy[0],0.55507246)

    def test_random_seed(self):
        pass

    def test_explicit_construction(self):
        data_arff = arff.Arff(self.arff_path, label_count=1)
        my_learner = baseline_learner.BaselineLearner(data_arff)
        session = manager.ToolkitSession(arff=self.arff_path, learner=my_learner, eval_method="training",
                                         eval_parameter=None, print_confusion_matrix=False, normalize=False,
                                         random_seed=None, label_count=1)


    def test_old_style_commands(self):
        my_learner = baseline_learner.BaselineLearner
        session1 = manager.ToolkitSession(arff=self.arff_path, learner=my_learner,
                                          eval_method="random", eval_parameter=.7, label_count=1)
        session2 = manager.ToolkitSession(arff=self.arff_path, learner=my_learner,
                                          eval_method="static", eval_parameter=self.arff_path, label_count=1)
        session3 = manager.ToolkitSession(arff=self.arff_path, learner=my_learner,
                                          eval_method="cross", eval_parameter=10, label_count=1)


    def test_learner_kwargs(self):
        """

        Returns:

        """
        my_learner = baseline_learner.BaselineLearner
        data_arff = arff.Arff(self.arff_path, label_count=1)
        session = manager.ToolkitSession(arff=self.arff_path, learner=my_learner, data=data_arff, example_hyperparameter=.5, label_count=1)
        self.assertEqual(session.learner.data_shape,(690,16))
        self.assertEqual(session.learner.example_hyperparameter, .5)


    def test_confusion_matrix(self):
        my_learner = baseline_learner.BaselineLearner
        data_arff = arff.Arff(self.arff_path, label_count=1)
        session = manager.ToolkitSession(arff=self.arff_path, learner=my_learner, data=data_arff, example_hyperparameter=.5, label_count=1)
        session.train()
        cm = session.learner.get_confusion_matrix(data_arff.get_features(), data_arff.get_labels())
        self.assertEqual('383',cm[-1,-1])

    def test_multidimensional_label(self):
        pass


if __name__=="__main__":
    print("here")
    suite = TestLoader().loadTestsFromTestCase(TestManager)
    TextTestRunner(verbosity=2).run(suite)
