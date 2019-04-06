from toolkit import baseline_learner
from toolkit import manager, arff
from toolkit import graph_tools
import matplotlib.pyplot as plt
import numpy as np
from toolkit.manager import Arff

arff_path = r"./test/datasets/creditapproval.arff"

## Read in arff file
credit_approval = arff.Arff(arff=arff_path, label_count=1)

## Example usage

# Get 1st row of features as an ARFF
features = credit_approval.get_features(slice(0,1))

# Print as arff
print(features)

# Print Numpy array
print(features.data)

# Print first row as Numpy array
print(features[0, :])

# Get all labels as numpy array using slicing
labels = credit_approval.get_labels()[:]

# Manual Training/Test
my_learner = baseline_learner.BaselineLearner
session = manager.ToolkitSession(arff=credit_approval, learner=my_learner)
train_features, train_labels, test_features, test_labels = session.training_test_split(.7)  # 70% training
session.train(train_features, train_labels)
session.test(test_features, test_labels)
print(session.training_accuracy)

# Pass on hyperparameters to learner
session = manager.ToolkitSession(arff=credit_approval, learner=my_learner, data=credit_approval, example_hyperparameter=.5)
print(session.learner.example_hyperparameter) # .5

# Automatic
session2 = manager.ToolkitSession(arff=credit_approval, learner=my_learner, eval_method="random", eval_parameter=.7)

# Cross validate
session3 = manager.ToolkitSession(arff=credit_approval, learner=my_learner)
session3.cross_validate(folds=10, reps=3)
print(session3.test_accuracy)

# Print Confusion matrix
cm = session3.learner.get_confusion_matrix(credit_approval.get_features(), credit_approval.get_labels())
print(cm)

## Graph a function
y_func = lambda x: 5 * x**2 + 1 # equation of a parabola
x = np.linspace(-1, 1, 100)
plt.plot(x, y_func(x))
plt.show()

## Scatter plot with 2 variables with labels coloring
x = credit_approval[:,1]
y = credit_approval[:,2]
labels = credit_approval[:, -1]
graph_tools.graph(x=x, y=y, labels=labels, xlim=(0,30), ylim=(0,30), title="Credit Approval")
