from __future__ import (absolute_import, division, print_function, unicode_literals)
from unittest import TestCase,TestLoader,TextTestRunner
from toolkit import baseline_learner, utils, manager, arff
import numpy as np
import os
from toolkit import utils
import subprocess

class TestManager(TestCase):

    infinity = float("infinity")

    def setUp(self):

        ## Download .arff data
        self.iris_data = "./datasets/iris.arff"
        iris_url = "http://axon.cs.byu.edu/data/uci_class/iris.arff"
        utils.save_arff(iris_url, self.iris_data)

    def test_commandline_from_python(self):
        ## Create manager - from commandline argument
        args = r'-L baseline -A {} -E training -V'.format(self.iris_data)
        my_manager = manager.MLSystemManager()
        session = my_manager.create_session_from_argv(args)
        assert session.arff.data[0][0] == 5.1
        self.assertAlmostEqual(session.training_accuracy[0],1/3)

    def test_commandline_error(self):
        ## Create manager - from commandline argument
        # args = r'BAD COMMAND'.format(self.iris_data)
        # my_manager = manager.MLSystemManager()
        # with self.assertRaises(Exception) as context:
        #     session = my_manager.create_session_from_argv(args)
        #self.assertTrue('Error' in str(context.exception))
        pass


    def test_commandline_from_subprocess(self):
        args = r'-L baseline -A {} -E training -V'.format(self.iris_data)
        child = subprocess.Popen([r"python", args], shell=False)
        print(child.communicate()[0],child.returncode)
        #subprocess.popen()

if __name__=="__main__":
    suite = TestLoader().loadTestsFromTestCase(TestManager)
    TextTestRunner(verbosity=2).run(suite)
