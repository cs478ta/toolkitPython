from __future__ import (absolute_import, division, print_function, unicode_literals)
from .arff import Arff
import math
import numpy as np
import warnings
#from sklearn.metrics import confusion_matrix
# this is an abstract class


class SupervisedLearner:

    def train(self, features, labels):
        """
        Before you call this method, you need to divide your data
        into a feature matrix and a label matrix.
        :type features: Arff
        :type labels: Arff
        """
        raise NotImplementedError()

    def predict_all(self, features):
        """
        A feature vector goes in. A label vector comes out. (Some supervised
        learning algorithms only support one-dimensional label vectors. Some
        support multi-dimensional label vectors.)
        :type features: [float]
        :type labels: [float]
        """
        raise NotImplementedError()

    def check_shape(self, arr1, arr2):
        try:
            assert arr1.shape[0]==arr2.shape[0] # must have same number of instances
        except:
            raise Exception("Arrays must have same dimension along row axis; shape mismatch {} {}".format(arr1.shape[0],arr2.shape[0]))

    def measure_accuracy(self, features, labels, eval_method=None):
        """
        The model must be trained before you call this method. If the label is nominal,
        it returns the predictive accuracy. If the label is continuous, it returns
        the mean squared error (MSE).

        Args:
            features (Arff, array-like):
            labels (Arff, array-like):

        Returns:
            float
        """
        self.check_shape(features, labels)

        if eval_method==None:
            if isinstance(labels, Arff):
                # Check first label, if nominal, measure accuracy
                if labels.is_nominal():
                    eval_method = "accuracy"
                else:
                    eval_method = "mse"
            elif isinstance(labels, np.ndarray):
                warnings.warn("Numpy array passed with no evaluation method, measuring accuracy")
                eval_method = "accuracy"

        if eval_method == "mse":
            return self.calc_mse(features, labels)
        elif eval_method == "accuracy":
            return self.calc_accuracy(features, labels)

    def calc_mse(self, features, labels):
        self.check_shape(features, labels)
        feat = features
        targ = labels
        pred = np.asarray(self.predict_all(feat))
        delta = targ - pred
        sse = np.sum(delta**2)
        return sse/features.shape[0]

    def calc_accuracy(self, features, labels, return_scalar=True):
        """ Calculates accuray. Supports multiple output/label dimensions. Returns an accuracy for each output.

        Args:
            features (array-like, Arff):
            labels (array-like, Arff):
            return_scalar (bool): Return float (rather than ndarray); will return ndarray if multiple dimensional output
        Returns:
            Array of accuracies (one for each output dimension)
        """
        self.check_shape(features, labels)
        feat = features

        if isinstance(labels, Arff):
            targ = (labels.data).astype(int)
        else:
            targ = (labels).astype(int)

        pred = np.asarray(self.predict_all(feat)).astype(int)
        accuracy = np.sum(targ==pred, axis=0)/features.shape[0]

        if return_scalar:
            if accuracy.size==1:
                [accuracy] = accuracy
            else:
                warnings.warn("return_scalar=True, but accuracy is a vector; returning accuracy as nd_array")

        return accuracy

    def get_confusion_matrix(self, features, labels):
        """ Get confusion matrix from features, labels

        Args:
            features:
            labels:

        Returns:
            confusion_matrix (np.ndarray)
        """
        # Get label names
        label_unique_values=[]
        if isinstance(labels, Arff) and len(labels.enum_to_str) > 0 and labels.enum_to_str[-1] != {}:
            label_dict = labels.enum_to_str[-1] # get dictionary for last column
            for i in range(0, len(label_dict)):
                label_unique_values.append(label_dict[i]) # make sure in numerical order
            label_unique_values = np.asarray(label_unique_values)
        else:
            label_unique_values = None

        ## Prep/reshape
        pred = np.asarray(self.predict_all(features)).reshape(-1)
        labels = labels.reshape(-1)
        self.check_shape(features, labels)

        ## Get confusion matrix
        cm = self.confusion_matrix(y_true=labels, y_pred=pred, labels=label_unique_values)

        ## Prep to output - add label values
        if not label_unique_values is None:
            top_row = np.r_[[""], label_unique_values].reshape(1,-1)
            p=np.c_[label_unique_values, cm]
            cm = np.r_[top_row, p]
        return cm

    def confusion_matrix(self, y_true, y_pred, labels=None, sample_weight=None):
        """ Get confusion matrix from labels and predictions; mostly stolen from sci-kit learn

        Args:
            y_true (array-like):
            y_pred (array-like):
            labels (array-like): list/array of labels, only needed for size?

        Returns:
            confusion_matrix (np.ndarray)
        """

        from scipy.sparse import coo_matrix

        if sample_weight is None:
            sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
        else:
            sample_weight = np.asarray(sample_weight)

        labels = np.asarray(labels)
        if labels is None:
            labels = np.unique(np.r_[y_true, y_pred])

        n_labels = labels.size

        # intersect y_pred, y_true with labels, eliminate items not in labels
        ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
        y_pred = y_pred[ind]
        y_true = y_true[ind]

        # also eliminate weights of eliminated items
        sample_weight = sample_weight[ind]

        # Choose the accumulator dtype to always have high precision
        if sample_weight.dtype.kind in {'i', 'u', 'b'}:
            dtype = np.int64
        else:
            dtype = np.float64

        CM = coo_matrix((sample_weight, (y_true, y_pred)),
                        shape=(n_labels, n_labels), dtype=dtype,
                        ).toarray()
        return CM