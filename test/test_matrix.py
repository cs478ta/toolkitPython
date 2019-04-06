from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner
from toolkit.arff import Arff
import numpy as np
import os
from toolkit import utils

class TestMatrix(TestCase):

    infinity = float("inf") # using NaN messes up equality testing

    def setUp(self):

        # NOTE: for discrete attributes, at least one value must be a float in order for numpy array
        # functions to work properly.
        data = np.array([[1.5, -6, 1.0],
                         [2.3, -8, 2],
                         [4.1, self.infinity, 2]])
        m = Arff(data, label_count=1)
        m.attr_names = ['A', 'B', 'C']
        m.str_to_enum = [{}, {}, {'R': 0, 'G': 1, 'B': 2}]
        m.enum_to_str = [{}, {}, {0: 'R', 1: 'G', 2: 'B'}]
        self.m = m

        data2 = np.array([[0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.1, 1.1, 2.1, 3.1, 1.0],
                   [0.2, 1.2, 2.2, 3.2, 1.0],
                   [0.3, 1.3, 2.3, 3.3, 2.0],
                   [0.4, 1.4, 2.4, 3.4, 2.0]])

        m2 = Arff(data2, label_count=1)
        m2.attr_names = ['A', 'B', 'C', 'D', 'E']
        m2.str_to_enum = [{}, {}, {}, {}, {'R': 0, 'G': 1, 'B': 2}]
        m2.enum_to_str = [{}, {}, {}, {}, {0: 'R', 1: 'G', 2: 'B'}]
        self.m2 = m2

        self.credit_data_path = os.path.join(utils.get_root(),"test/datasets/creditapproval.arff")
        self.iris_path = os.path.join(utils.get_root(),"test/datasets/iris.arff")

    def test_create_subset_arff(self):
        m2 = Arff(self.m2, [1,2], slice(1,3))
        self.assertEqual(m2.shape, (2,2))

        m2 = Arff(self.m2, slice(1,3), [1,2])
        self.assertEqual(m2.shape, (2,2))

        # Automatic label inference
        self.m2.label_count=3
        m2 = Arff(self.m2, slice(1,3), slice(1,None), label_count=None)
        self.assertEqual(3, m2.label_count)
        m2 = Arff(self.m2, slice(1,3), slice(1,-1), label_count=None)
        self.assertEqual(2, m2.label_count)

    def test_copy_and_slice(self):
        d = Arff(self.credit_data_path)
        e = d.copy()

        e._copy_and_slice_arff(d, 1, 5)
        self.assertEqual(e.shape, (1,1))

        e._copy_and_slice_arff(d, slice(1,4), slice(2,4))
        self.assertEqual(e.shape, (3, 2))

        # This will create a 1D array, returning coords (1,1), (2,5), (3,7)
        e._copy_and_slice_arff(d, [1,2,3,7], [1,5,7,8])
        self.assertEqual(e.shape, (4, ))

        e._copy_and_slice_arff(d, [1,2,3,7], slice(0,5))
        self.assertEqual(e.shape, (4, 5))


    def test_arff_constructor(self):
        """ Tests construction of Arff from path, arff, numpy array
        """
        ## Create a Matrix object from arff
        credit = Arff(arff=self.credit_data_path)
        credit3 = Arff(arff=credit)
        credit2 = Arff(arff=credit.data)

        np.testing.assert_array_almost_equal(credit.data, credit2.data)
        np.testing.assert_array_almost_equal(credit2.data, credit3.data)

    def test_set_size(self):
        m = Arff()
        m.set_size(3, 4)
        self.assertEqual(m.shape[0], 3)
        self.assertEqual(m.shape[1], 4)

    def test_download(self):
        # Test download
        url = "http://axon.cs.byu.edu/data/uci_class/iris.arff"
        if os.path.exists(self.iris_path):
            os.remove(self.iris_path)
        utils.save_arff(url, self.iris_path)

    def test_load_arff(self):
        """ Tests downloading and loading arff file
        """

        t = Arff()
        t.load_arff(self.iris_path)
        self.assertListEqual(t.data[t.shape[0]-1].tolist(), [5.9, 3.0, 5.1, 1.8, 2.0])

    def test_rows(self):
        self.assertEqual(self.m.shape[0], 3)

    def test_cols(self):
        self.assertEqual(self.m.shape[1], 3)

    def test_row(self):
        self.assertListEqual(self.m.data[1].tolist(), [2.3, -8, 2])

    def test_col(self):
        self.assertListEqual(self.m.data[:,1].tolist(), [-6, -8, self.infinity])

    def test_get(self):
        self.assertEqual(self.m.data[0, 2], 1)
        self.assertEqual(self.m.data[2, 0], 4.1)

    # def test_set(self):
    #     self.m.set(2, 1, 2.5)
    #     self.assertEqual(self.m.get(2, 1), 2.5)

    def test_attr_name(self):
        print(type(self.m))
        name = self.m.attr_name(2)
        self.assertEqual(name, 'C')

    def test_set_attr_name(self):
        self.m.set_attr_name(2, 'Color')
        self.assertEqual(self.m.attr_name(2), 'Color')

    def test_attr_value(self):
        self.assertEqual(self.m.attr_value(2, 0), 'R')

    def test_value_count(self):
        self.assertEqual(self.m.unique_value_count(1), 0)     # continuous
        self.assertEqual(self.m.unique_value_count(2), 3)     # R, G, B

    def test_shuffle(self):
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        pass

    def test_column_mean(self):
        self.assertAlmostEqual(self.m.column_mean(0), 2.6333, 4)
        self.assertAlmostEqual(self.m.column_mean(1), -7, 4)

    def test_column_min(self):
        self.assertEqual(self.m.column_min(0), 1.5)
        self.assertEqual(self.m.column_min(1), -8)

    def test_column_max(self):
        self.assertEqual(self.m.column_max(0), 4.1)
        self.assertEqual(self.m.column_max(1), -6)

    def test_most_common_value(self):
        self.assertEqual(self.m.most_common_value(0), 1.5)
        self.assertEqual(self.m.most_common_value(2), 2)

    def test_append_rows(self):
        test_matrix = Arff(self.m)

        # Verify it works with other matrices
        test_matrix.append_rows(test_matrix)
        assert test_matrix.data.shape == (6,3)

        # Verify it works with numpy array
        test_matrix.append_rows(test_matrix.data)
        assert test_matrix.data.shape == (12,3)

        # Verify it works with 2D list
        test_matrix.append_rows(test_matrix.data)
        assert test_matrix.data.shape == (24,3)

        # Verify incompatible number of rows
        with self.assertRaises(Exception) as context:
            test_matrix.append_rows(self.m.data[:,:2])
        print(str(context.exception))
        self.assertTrue('Incompatible number of columns' in str(context.exception))


    def test_append_columns(self):
        test_matrix = Arff(self.m)

        # Verify it works with other matrices
        test_matrix.append_columns(test_matrix)
        assert test_matrix.data.shape == (3,6)

        # Verify it works with numpy array
        test_matrix.append_columns(test_matrix.data)
        assert test_matrix.data.shape == (3,12)

        # Verify it works with 2D list
        test_matrix.append_columns(test_matrix.data)
        assert test_matrix.data.shape == (3,24)

        # Verify incompatible number of columns
        with self.assertRaises(Exception) as context:
            test_matrix.append_columns(self.m.data[:1,:])
        self.assertTrue('Incompatible number of rows' in str(context.exception))


    def test_get_features(self):
        """ Tests construction of Arff from path, arff, numpy array
        """
        # Create a Matrix object from arff
        credit = Arff(arff=self.credit_data_path, label_count=1)
        credit.label_count=0
        np.testing.assert_equal(credit.data, credit.get_features().data)

        ## Test label inference
        credit.label_count = 5
        self.assertEqual(credit.get_labels().shape, (690, 5))

        ## Copy last 8 columns
        credit2 = Arff(credit, col_idx=slice(-8, None))
        self.assertEqual(credit2.label_count, 5)
        self.assertEqual((690,3), credit2.get_features().shape)

        ## Verify 0 labels
        credit.label_count = 0
        self.assertEqual((690, 16), credit.get_features().shape)
        self.assertEqual((690, 0), credit.get_labels().shape)

if __name__=="__main__":
    #my_test = TextMatrix.test_get_features
    suite = TestLoader().loadTestsFromTestCase(TestMatrix.test_get_features)
    TextTestRunner(verbosity=2).run(suite)
