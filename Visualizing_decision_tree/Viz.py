import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()

test_idx = [0, 50, 100]

# Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Train model
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(train_target)
print(clf.predict(test_data))

plt.figure(figsize=(12, 8))
tree.plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.show()

print(test_data[1], test_target[1])

print(iris.feature_names, iris.target_names)