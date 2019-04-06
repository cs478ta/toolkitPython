## Python Toolkit Tutorial

This short tutorial demonstrates some basic functionality of the Python toolkit for CS 478.

### Modules

First, import the modules you will be using.
```
from toolkit import baseline_learner, manager, arff
import numpy as np
```

### Intro to Numpy Arrays
Numpy is Python's premier numerical array module. While Numpy arrays handle n-dimensions, this toolkit is tailored for .arff file data, which is generally 2-dimensional. The `.shape` property of a Numpy array will return a tuple of the array dimensions (rows, columns).

The array can also be "sliced" to obtain subsets:
```
my_array = np.asarray(range(0,25)).reshape(5,5)

# Get first two rows, from 4th column to the end
my_array[0:2, 3:]

# Get every other row, start at last column and go backward
my_array[::2, -1::-1]

# Get indices for all rows that have a 5 in them
row_idx = np.where(my_array==5)[0]
my_array[row_idx]

```

### Arff object class
Most of the datasets we used are stored in an .arff file format. The toolkit can create Python representations of these files:

```
arff_path = r"./test/datasets/creditapproval.arff"
credit_approval = arff.Arff(arff=arff_path, label_count=1)
```

Here, `credit_approval` is an Arff object. The Arff object is mostly a wrapper around a 2D numpy array, which is stored as the 'data' Arff class variable, i.e. `credit_approval.data`. The Arff object also contains all the information needed to recreate the Arff file, including feature names, the number of columns that are considered "outputs" (labels), whether each feature is nominal or continuous, and the list of possible values for nominal features. Note that:

* The Arff object automatically encodes nominal/string features as integers. 
* The toolkit presently supports multiple labels, which are assumed to be the rightmost column(s). The number of labels should be passed explicitly with `label_count`, which is typically 1.
* `print(credit_approval)` will print the object as Arff text. Alternatively, a .arff style string can be obtained by taking `str(credit_approval)`.

The Arff object can also be sliced like traditional numpy arrays. E.g., the first row of data as a numpy array would be:

```
credit_approval[0,:]
```

Note that slicing this way returns a numpy 2D array, not an Arff object. To create a new Arff object that has been sliced, one can use:

```
# Get first 10 rows, first 3 columns
new_arff = Arff(credit_approxal, row_idx = slice(0,10), col_idx=slice(0,3), label_count=1)
```

Alternatively, one can use a `list` or `int` for either the `col_idx` or `row_idx`, but they should not be used for both simultaneously:

```
# Get rows 0 and 2, columns 0 through 9
new_arff = Arff(credit_approxal, row_idx = [0,2], col_idx=slice(0,10), label_count=1)

# Get row 1, all columns
new_arff = Arff(credit_approxal, row_idx = 1, label_count=1)

# Don't do this
new_arff = Arff(credit_approxal, row_idx = [2,3,8], col_idx = [1,2,3], label_count=1)
```

This ```new_Arff``` object will should copy the numpy array data underlying the original Arff. ```Arff.copy()``` can also be used to make a safe, deep copy of an Arff object.

To get the features of an Arff object as another Arff object, one can simply call:
```credit_approval.get_features()```

Similarly, for labels:
```credit_approval.get_labels()```

This may be helpful, since the Arff object has methods like:
* `unique_value_count(col)`: Returns the number of unique values for nominal variables
* `is_nominal(col)`: Returns true if the column is nominal
* `shuffle(buddy=None)`: Shuffles the data; supplying a buddy Arff with the same number of rows will shuffle both objects in the same order.

#### Other examples:
```
# Get 1st row of features as an ARFF
features = credit_approval.get_features(slice(0,1))

# Print as arff
print(features)

# Print Numpy array
print(features.data)

# Get shape of data: (rows, columns)
print(features.shape)

```

### Creating Learners

See the ```baseline_learner.py``` and its `BaselineLearner` class for an example of
the format of the learner. Learning models should inherit from the `SupervisedLearner` base class and override
the `train()` and `predict()` functions. It should probably also have a constructor, i.e. `def __init__(self, argument1, argument2):` that can be used to initialize learner weights, hyperparameters, etc.

### Training
Session objects can be created to facilitate training. Session objects, at a minimum, should be passed 1) an Arff file path or Arff object and 2) uninstantiated learner class.

```
# Create arff object
credit_approval = arff.Arff(arff=arff_path, label_count=1)

# Declare learner
my_learner = baseline_learner.BaselineLearner

# Create session
session = manager.ToolkitSession(arff=credit_approval, learner=my_learner)

# Split training/test data
train_features, train_labels, test_features, test_labels = session.training_test_split(.7)  # 70% training

# session.train mostly calls learner.train() and records accuracy
session.train(train_features, train_labels)
session.test(test_features, test_labels)

# Session keeps track of accuracy by epoch
print(session.training_accuracy)
print(session.test_accuracy)
```

Because the learner is not instantiated, you can pass named learner arguments to the session, which will be passed on to the learner. For instance, you might want to initialize a learner object with a certain learning rate, or pass the Arff object when it is instantiated to initialize an appropriately sized weight matrix.

```
# Pass on hyperparameters to learner
session = manager.ToolkitSession(arff=credit_approval, learner=my_learner, data=credit_approval, example_hyperparameter=.5)
print(session.learner.data_shape, (690, 16))
print(session.learner.example_hyperparameter, .5)
```

The toolkit also natively supports cross-validation:

```
# Cross-validate, 10 folds, perform 3x
session3 = manager.ToolkitSession(arff=credit_approval, learner=my_learner)
session3.cross_validate(folds=10, reps=3)
print(session3.test_accuracy)
```

It can also create a confusion matrix (e.g. for nominal labels):

```
# Print Confusion matrix
cm = session3.learner.get_confusion_matrix(credit_approval.get_features(), credit_approval.get_labels())
print(cm)
```

### End-to-end training
The toolkit also supports end-to-end training, for automatic training, training/testing, and cross-validated training, similar to the C++ and Java versions of the toolkit. The `eval_method` and `eval_parameter` are the same as they are in the commandline variant and the C++ and Java toolkits.

```
# Train with random 70% split, test with other 30% of data
session2 = manager.ToolkitSession(arff=credit_approval, learner=my_learner, eval_method="random", eval_parameter=.7)
```

### Graphing
A tiny graphing wrapper around matplotlib is included. See ```graph_tools.py```.

```
from toolkit import graph_tools
import matplotlib.pyplot as plt
import numpy as np

arff_path = r"./test/datasets/creditapproval.arff"
credit_approval = arff.Arff(arff=arff_path, label_count=1)


## Graph a function using matplotlib
y_func = lambda x: 5 * x**2 + 1 # equation of a parabola
x = np.linspace(-1, 1, 100)
plt.plot(x, y_func(x))
plt.show()

## Scatter plot with 2 variables with labels coloring using graph_tools.py
x = credit_approval[:,1]
y = credit_approval[:,2]
labels = credit_approval[:, -1]
graph_tools.graph(x=x, y=y, labels=labels, xlim=(0,30), ylim=(0,30))
```

![alt text](https://raw.githubusercontent.com/cs478ta/CS478.github.io/master/toolkitPython/Scatter.png)

### Command-line:
Training and testing can be performed via command-line (as in the Java and C++ toolkits).
Running the toolkit from command-line will require modifying the toolkit to 1) import your learner and 2) correctly parse the `learningAlgorithm` argument to instantiate your learner class.

As example, execute the following commands from the root directory of this
repository.

```bash
mkdir datasets
wget http://axon.cs.byu.edu/~martinez/classes/478/stuff/iris.arff -P datasets/
python -m toolkit.manager -L baseline -A datasets/iris.arff -E training
```

Notice that you must specify the module "toolkit" as well as the manager file. 
Aside from this difference, commands follow the same syntax as the other toolkits.

For information on the expected syntax, you may run:

```bash
python -m toolkit.manager --help
```

#### Command-line Usage: 
```
python -m toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E [EvaluationMethod] {[ExtraParamters]} -N {-R [seed]}

Example:
python -m toolkit.manager -L baseline -A ./test/datasets/iris.arff -E training

Possible evaluation methods are:

# Only train
python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E training

# Train and test, with test data in a separate arff file
python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E static [TestARFF_File]

# Randomly split data into train/test, then train, then test
python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E random [PercentageForTraining]

# Cross-validate
python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]

Other options:
-R [int]: random seed (should be set for reproducible results)
-N : normalize continuous variables
```
