from sklearn import tree

features = [[300, 2], [200, 8], [450, 2], [150, 9]]

labels = [1, 0, 1, 0]

clf_car = tree.DecisionTreeClassifier()

cls_car = clf_car.fit(features, labels)

print(clf_car.predict([[100, 9]]))