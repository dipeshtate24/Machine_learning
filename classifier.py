import random
from sklearn import tree
from sklearn import datasets
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    
    def predict(self, X_test):
        prediction = []
        for row in X_test:
            # label = random.choice(self.y_train)
            label = self.closest(row)
            prediction.append(label)
        return prediction
    
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        
        return self.y_train[best_index]
    # import dataset

iris = datasets.load_iris()

X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5)


my_classifier = tree.DecisionTreeClassifier()


    # from sklearn.neighbors import KNeighborsClassifier

my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

prediction = my_classifier.predict(X_test)

print(accuracy_score(y_test, prediction))
