from sklearn import metrics
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main(unused_argv):
    # Load Dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    # Build DNN
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(X_train, y_train, epochs=200, verbose=1)

    # Predict
    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=1)

    # Accuracy
    score = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: {:.4f}".format(score))


if __name__ == "__main__":
    main(None)