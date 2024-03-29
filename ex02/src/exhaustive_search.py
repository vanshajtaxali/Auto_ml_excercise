import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier


def plot_function(K_values: List[int], cv_scores: Dict[str, List[float]]) -> None:
    """
    Plot the cross-validation accuracy for each distance metric and K value.

    :K_values: list of K values
    :cv_scores: dictionary with distance as key and list of accuracies as values
    """
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    for i, (d, accuracies) in enumerate(cv_scores.items()):
        assert len(accuracies) == len(K_values)
        ax[i].plot(K_values, accuracies)
        ax[i].set_title('Distance metric: ' + d)
        ax[i].set_xlabel('number of Neighbors K')
        ax[i].set_ylabel('CV Accuracy')
        ax[i].grid(True)

    plt.savefig('plots.png')
    plt.show()


def main(test_portion: float) -> None:
    """
    Main method which loads the iris dataset, splits it into train/test sets
    and finds the best values for the d (distance metric) and K hyperparameters.

    :test_portion: fraction of data we want to use for testing
    """
    # import the iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_portion,
                                                        random_state=28)

    # Hyperparameters K and d.
    K_values = [k for k in range(1, int(0.9*len(y_train))-1)]
    distance_metrics = ['euclidean', 'manhattan']

    # dictionary that will hold the cross-validation results for each metric
    cv_scores = {key: [] for key in distance_metrics}

    # 10-fold cross-validation accuracy for each combination of d and K
    for d in distance_metrics:
        for k in K_values:
            cv_score = cross_val_score(
                KNeighborsClassifier(n_neighbors=k, metric=d),
                X_train,
                y_train,
                cv=10
            )
            cv_scores[d].append(cv_score.mean())

    # TODO: Select the best configurations based on the cross-validation accuracy 
    #       (it might be more than one) and add them to optimal_configs. optimal_configs
    #       should be a list of lists with their first element being the K value and 
    #       the second one the distance d of the best configuration.
    #       E.g. optimal_config = [[38, 'euclidean'], [3, 'manhattan']]
    optimal_configs = []
    for d in distance_metrics:
        values = np.array(cv_scores[d])
        max_ = values.max()
        for i in np.argwhere(values == max_):
            optimal_configs.append([i[0], d])

    # Report the performance of each optimal config of KNN on the test set
    for k, d in optimal_configs:
        knn = KNeighborsClassifier(n_neighbors=k, metric=d)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info('\nThe accuracy of KNN with (K, d) = (%d, %s) is %.3f.', k, d, accuracy)

    plot_function(K_values, cv_scores)


if __name__=='__main__':
    cmdline_parser = argparse.ArgumentParser('KNN Classifier')

    # Optional args
    cmdline_parser.add_argument('-p',
                                default=0.33,
                                type=float,
                                help="test portion")
    cmdline_parser.add_argument('-v',
                                '--verbose',
                                action='store_true',
                                default=False,
                                help="verbosity level")

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(test_portion=args.p)


