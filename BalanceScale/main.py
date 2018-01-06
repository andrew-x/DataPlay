import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier


def load_data():
    """
    loading csv to pandas dataframe

    :return: dataframe
    """
    return pd.read_csv('data/data.txt', sep=',', header=0)


def prepare_data(data):
    """
    vectorize balance

    :return: data with vectorized classes
    """
    data['balance'] = LabelEncoder().fit_transform(data['balance'])
    return data


def explore_data(data):
    """
    perform various forms of data visualisation and analysis

    :param data:
    :return: None
    """
    [m, n] = data.shape
    attributes = data.columns.values

    def data_summary():
        """
        Prints summary of data

        :return: None
        """
        print('Number of rows: {}'.format(m))
        for i in range(0, 3):
            [m_class, _] = data[data['balance'] == i].shape
            print('Number of class {}: {} | {}%'.format(i, m_class, m_class/m * 100))
        print(data.info())
        print(data.describe())

    def occurrence_histogram():
        """
        displays histogram of feature value occurrences.

        :return: None
        """
        nrows = n // 2
        fig, ax = plt.subplots(ncols=2, nrows=nrows)
        for i in range(1, len(attributes)):
            row, col = (i-1)//2, 1 if i % 2 == 0 else 0
            ax_ref = ax[col] if nrows <= 1 else ax[row, col]
            sns.distplot(data[attributes[i]], kde=False, vertical=True, ax=ax_ref)
            ax_ref.set(xlabel='# of occurrences', ylabel='value', title=attributes[i])
        fig.subplots_adjust(hspace=0.75, wspace=0.75)
        plt.show()

    def correlation_matrix():
        """
        Show matrix of correlations

        :return: None
        """
        plt.subplots(figsize=(10, 10))
        sns.heatmap(data[attributes].corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.title('Correlation Matrix')
        plt.show()

    def covariance_matrix():
        """
        Show matrix of covariances between attributes.

        :return: None
        """
        plt.subplots(figsize=(10, 10))
        sns.heatmap(data[attributes].cov(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.title('Covariance Matrix')
        plt.show()

    def t_sne_scatter():
        """
        Shows a scatter plot of T-SNE embed

        :return: None
        """
        features = data.drop(['balance'], axis=1)
        labels = data['balance']

        t_sne = TSNE(n_components=2)
        embed = t_sne.fit_transform(features)
        x1, y1, x2, y2, x3, y3 = [], [], [], [], [], []
        for i, e in enumerate(embed):
            if labels[i] == 1:
                x1 += [e[0]]
                y1 += [e[1]]
            elif labels[i] == 2:
                x2 += [e[0]]
                y2 += [e[1]]
            else:
                x3 += [e[0]]
                y3 += [e[1]]
        plt.scatter(x1, y1, c='r')
        plt.scatter(x2, y2, c='b')
        plt.scatter(x3, y3, c='g')
        plt.title('T-SNE Embedding')
        plt.show()

    def pca_scatter():
        """
        Shows a scatter plot of PCA decomposition

        :return: None
        """
        features = data.drop(['balance'], axis=1)
        labels = data['balance']

        pca = PCA(n_components=2)
        decomposition = pca.fit_transform(features)
        x1, y1, x2, y2, x3, y3 = [], [], [], [], [], []
        for i, e in enumerate(decomposition):
            if labels[i] == 1:
                x1 += [e[0]]
                y1 += [e[1]]
            elif labels[i] == 2:
                x2 += [e[0]]
                y2 += [e[1]]
            else:
                x3 += [e[0]]
                y3 += [e[1]]
        plt.scatter(x1, y1, c='r')
        plt.scatter(x2, y2, c='b')
        plt.scatter(x3, y3, c='g')
        plt.title('PCA Decomposition')
        plt.show()

    data_summary()
    occurrence_histogram()
    correlation_matrix()
    covariance_matrix()
    t_sne_scatter()
    pca_scatter()


def engineer_data(data):
    """
    Returns modified version of data with left and right aggregate features while dropping weight and distance features

    :param data: data to work with
    :return: modified dataframe
    """
    data['left'] = data['left_weight'] * data['left_distance']
    data['right'] = data['right_weight'] * data['right_distance']
    data = data.drop(['left_weight', 'left_distance', 'right_weight', 'right_distance'], axis=1)
    return data


def model_data(data, test=False):
    """
    model the data and evaluate the selected model.

    :param data: all data to be used
    :param test: whether or not we are testing
    :return: None
    """
    features = data.drop(['balance'], axis=1)
    labels = data['balance'].to_frame()
    X_work, X_test, y_work, y_test = train_test_split(features, labels)
    # we don't touch the test set until the end.

    def grid_search(estimator, grid, X, y):
        gs = GridSearchCV(estimator, cv=5, n_jobs=-1, param_grid=grid)
        gs.fit(X, y)
        print(gs.best_params_)
        return gs.best_estimator_

    def support_vector(X, y):
        svc = SVC()
        grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        return grid_search(svc, grid, X, y)

    def random_forest(X, y):
        rfc = RandomForestClassifier(n_jobs=-1)
        grid = {
            'n_estimators': np.arange(5, 15),
            'criterion': ['gini', 'entropy']
        }
        return grid_search(rfc, grid, X, y)

    def knn(X, y):
        knc = KNeighborsClassifier()
        grid = {
            'n_neighbors': np.arange(1, 10)
        }
        return grid_search(knc, grid, X, y)

    def perceptron(X, y):
        mlp = MLPClassifier()
        grid = {
            'activation': ['identity', 'logistic', 'tanh', 'relu']
        }
        return grid_search(mlp, grid, X, y)

    def vote(X, y):
        estimators = [
            ('random_forest', random_forest(X, y)),
            ('knn', knn(X, y)),
            ('perceptron', perceptron(X, y))
        ]
        vc = VotingClassifier(estimators, n_jobs=-1)
        vc.fit(X, y)
        return vc

    if not test:
        avg_train, avg_validate = 0, 0
        skf = StratifiedKFold(n_splits=5)
        for train_idx, test_idx in skf.split(X_work, y_work):
            X_train, X_test, y_train, y_test = \
                X_work.iloc[train_idx], X_work.iloc[test_idx], y_work.iloc[train_idx], y_work.iloc[test_idx]
            model = vote(X_train, y_train)
            avg_train += accuracy_score(y_train, model.predict(X_train))
            avg_validate += accuracy_score(y_test, model.predict(X_test))
        print('train: {}'.format(avg_train/5))
        print('validate: {}'.format(avg_validate/5))
    else:
        def precision_recall(y, scores):
            """
            Plots precision recall curve. Not used in article.

            :param y: true y values
            :param scores: decision boundary scores
            :return:
            """
            for i in range(0, 3):
                y_class = y['balance'].apply(lambda x: 1 if x == i else 0)  # binarized for class
                scores_class = scores[:, i].flatten()
                precision, recall, _ = precision_recall_curve(y_class, scores_class)
                average_precision = average_precision_score(y_class, scores_class)
                print('average precision: {}'.format(average_precision))

                plt.step(recall, precision, color='b', alpha=0.2, where='post')
                plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
                    average_precision))
                plt.show()

        model = vote(X_work, y_work['balance'])
        # train_scores = model.decision_function(X_work)
        # precision_recall(y_work, train_scores)

        print('train: {}'.format(model.score(X_work, y_work)))
        print('test: {}'.format(model.score(X_test, y_test)))


if __name__ == '__main__':
    sns.set(color_codes=True)

    raw_data = load_data()
    prepared_data = prepare_data(raw_data)
    engineered_data = engineer_data(prepared_data)
    # explore_data(engineered_data)
    model_data(engineered_data, test=True)
