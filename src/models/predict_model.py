import pickle
import pandas as pd
from sklearn import metrics
import time
import os


def predict_model(dir, model, X_test, y_test):
    with open(dir, 'rb') as f:
        clf = pickle.load(f)
    t0 = time.time()
    predicted = clf.predict(X_test)
    result_file = os.path.abspath(os.path.join(__file__, "../../..")) + "\\results\\" + model + '.txt'

    with open(result_file, 'w+') as f:
        f.write(metrics.classification_report(y_test, predicted))
        f.write("Time taken: " + str(time.time() - t0))


def main():
    X_test = pd.read_csv(os.path.abspath(os.path.join(__file__, "../../..")) + "\\data\\trained\\X_test_set.csv")
    X_test = X_test.drop('Unnamed: 0.1', 1)
    y_test = pd.read_csv(os.path.abspath(os.path.join(__file__, "../../..")) + "\\data\\trained\\y_test_set.csv")
    pickle_dir = os.path.abspath(os.path.join(__file__, "../../..")) + "\\models\\pickle_results\\"

    for model in ['decision_tree', 'kNN', 'mlp', 'random_forest', 'svm', 'gradient_boost']:
        dir = pickle_dir + model + '.pkl'
        predict_model(dir, model, X_test, y_test)


if __name__ == "__main__":
    main()
