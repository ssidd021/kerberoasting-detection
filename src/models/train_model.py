import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, tree
from sklearn.pipeline import make_pipeline
import os.path
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier


def svm(X_train, X_test, y_train, y_test, to_save_dir):
    clf = make_pipeline(preprocessing.StandardScaler(), SVC(gamma=0.01, degree=3, kernel='rbf'))
    clf.fit(X_train, y_train)

    predicted_y = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_y).ravel()
    precision_score = tp / (tp + fp)
    recall_score = tp / (tp + fn)
    print(metrics.confusion_matrix(y_test, predicted_y))
    print(metrics.classification_report(y_test, predicted_y))

    report = metrics.classification_report(y_test, predicted_y, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(to_save_dir, 'svm.csv'))


def decision_tree(X_train, X_test, y_train, y_test, to_save_dir):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    predicted_y = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_y).ravel()
    precision_score = tp / (tp + fp)
    recall_score = tp / (tp + fn)
    print(metrics.confusion_matrix(y_test, predicted_y))
    print(metrics.classification_report(y_test, predicted_y))

    report = metrics.classification_report(y_test, predicted_y, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(to_save_dir, 'decision_tree.csv'))


def kNN(X_train, X_test, y_train, y_test, to_save_dir):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    predicted_y = clf.predict(X_test)

    print(metrics.confusion_matrix(y_test, predicted_y))
    print(metrics.classification_report(y_test, predicted_y))

    report = metrics.classification_report(y_test, predicted_y, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(to_save_dir, 'kNN.csv'))


def gradient_boost(X_train, X_test, y_train, y_test, to_save_dir):
    clf = GradientBoostingClassifier(n_estimators=25, learning_rate=1.0, max_depth=20, random_state=0)
    clf.fit(X_train, y_train)

    predicted_y = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_y).ravel()
    precision_score = tp / (tp + fp)
    recall_score = tp / (tp + fn)
    print(metrics.confusion_matrix(y_test, predicted_y))
    print(metrics.classification_report(y_test, predicted_y))

    report = metrics.classification_report(y_test, predicted_y, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(to_save_dir, 'gradient_boost.csv'))


def neural_net_mlp(X_train, X_test, y_train, y_test, to_save_dir):
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(14, 10 ), random_state=1, max_iter=300, warm_start=True)
    clf.fit(X_train, y_train)

    predicted_y = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_y).ravel()
    precision_score = tp / (tp + fp)
    recall_score = tp / (tp + fn)
    print(metrics.confusion_matrix(y_test, predicted_y))
    print(metrics.classification_report(y_test, predicted_y))

    report = metrics.classification_report(y_test, predicted_y, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(to_save_dir, 'mlp.csv'))


def encode_string_values(df):
    for f in df.columns:
        if df[f].dtype == 'object':
            lbl_enc = preprocessing.LabelEncoder()
            df[f] = lbl_enc.fit_transform(df[f].astype(str).values)
    return df


def main():
    trained_path = os.path.abspath(os.path.join(__file__, "../../..")) + "\\data\\trained\\"
    models_dir = os.path.abspath(os.path.join(__file__, "../../..")) + "\\models\\"
    dir = os.path.abspath(os.path.join(__file__, "../../..")) + "\\data\\datasets_with_features\\"
    frames = []

    attacks = ["apt_sim", "kerberoasting", "brute_force", "dc_shadow", "dc_sync",
               "golden_ticket", "password_spraying", "remote_process_injection", "normal_events"]

    for attack in attacks:
        attack_path = dir + attack + ".csv"
        frames.append(pd.read_csv(attack_path))

    result = pd.concat(frames)  # Combine all dataframes
    result = result.fillna(0)  # Fill in all NA values
    result = encode_string_values(result)  # Encode non float values

    X_train, X_test, y_train, y_test = train_test_split(result.drop('is_attack', 1), result['is_attack'],
                                                        test_size=0.25, random_state=42, shuffle=True)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    X_test.to_csv(trained_path + 'test_set.csv')

    svm(X_train, X_test, y_train, y_test, models_dir)
    kNN(X_train, X_test, y_train, y_test, models_dir)
    decision_tree(X_train, X_test, y_train, y_test, models_dir)
    gradient_boost(X_train, X_test, y_train, y_test, models_dir)
    neural_net_mlp(X_train, X_test, y_train, y_test, models_dir)


if __name__ == '__main__':
    main()