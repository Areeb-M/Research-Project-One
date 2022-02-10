from json import load
import numpy as np
import pandas
import pickle

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB

# This number is used as the random state where applicable
MAGIC_NUMBER = 2

def load_df(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def df_split(df):
    return df.drop(columns=['Performance Rank', 'Name']), df['Performance Rank']

def svm_rbf(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel='rbf', probability=True, C=3)
    clf.fit(x_train, y_train)

    return {
        "Name" : "SVM RBF",
        "Accuracy": clf.score(x_test, y_test)
    }

def svm_poly(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel='poly', probability=True, C=3)
    clf.fit(x_train, y_train)

    return {
        "Name" : "SVM Poly",
        "Accuracy": clf.score(x_test, y_test)
    }

def svm_linear(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel='linear', probability=True, C=3)
    clf.fit(x_train, y_train)

    return {
        "Name" : "SVM Linear",
        "Accuracy": clf.score(x_test, y_test)
    }

def gauss_nb(x_train, x_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)

    return {
        "Name" : "Gaussian Naive Bayes",
        "Accuracy": clf.score(x_test, y_test)
    }

def complement_nb(x_train, x_test, y_train, y_test):
    clf = BernoulliNB()
    clf.fit(x_train, y_train)

    return {
        "Name" : "Bernoulli Naive Bayes",
        "Accuracy": clf.score(x_test, y_test)
    }

def mlp_classifier(x_train, x_test, y_train, y_test):
    clf = MLPClassifier(max_iter=1000)
    clf.fit(x_train, y_train)

    return {
        "Name" : "MLP Classifier",
        "Accuracy": clf.score(x_test, y_test)
    }

def rf_classifier(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=MAGIC_NUMBER)
    clf.fit(x_train, y_train)

    importances = {}
    for n, v in enumerate(clf.feature_importances_):
        importances[x_train.columns[n]] = v

    importances.update({
        "Name": "Random Forest Classifier",
        "Accuracy": clf.score(x_test, y_test)
    })

    return importances

def dt_classifier(x_train, x_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=MAGIC_NUMBER)
    clf.fit(x_train, y_train)

    importances = {}
    for n, v in enumerate(clf.feature_importances_):
        importances[x_train.columns[n]] = v

    importances.update({
        "Name": "Decision Tree Classifier",
        "Accuracy": clf.score(x_test, y_test)
    })

    return importances

def train_test_process(data, test_size):
    model_data  = train_test_split(data[0], data[1], test_size=test_size, random_state=MAGIC_NUMBER)
    models = [svm_rbf, svm_linear, svm_poly, gauss_nb, complement_nb, mlp_classifier, rf_classifier, dt_classifier]

    results = [model(*model_data) for model in models]
    dfs = [pandas.DataFrame(result, index=[0]) for result in results]
    df = pandas.concat(dfs)

    return df
    # results for each method produce a dictionary
    # method name, accuracy, dictionary of any other important metrics 
    # dictionaries are turned into DFs then concatenated

def main(df_name):
    prediction_df = load_df(df_name)
    data = df_split(prediction_df)
    split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for ratio in split_ratios:
        results = train_test_process(data, ratio)
        print(results)
        results.to_csv(f"results/{df_name}-{ratio}.csv")

if __name__ == '__main__':
    df_path = 'prediction_w_sentiment_df.pickle'
    main(df_path)