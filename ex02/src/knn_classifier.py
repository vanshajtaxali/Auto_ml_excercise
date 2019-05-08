import logging
import argparse
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets


class KNN(object):
    """
    Base class for the k-nearest neighbors.
    """
    def __init__(self, K: int) -> None:
        self.K = K

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        This method actually does not fit anything. It just keeps track of
        training examples so it can use them for prediction. You do not have
        to change anything here. It might be useful when you do cross-validation,
        since you use different training subsets for each fold.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: np.ndarray, distance: str) -> np.ndarray:
        """
        Method used to predict the class of test examples.
        Here you should implement the distance with respect
        to all other examples in the training set and estimate
        the class based on the most frequent class of the K
        nearest neighbours.

        :X_test: np.ndarray of test examples
        :distance: distance metric. Either 'euclidean' or 'manhattan'
        :return: np.ndarray of predicted classes
        """
        # Shapes:
        # X_test=(pred, features)
        # X_train=(train, features)
        # diff=(pred, train, features)
        # Calculate the difference of the features
        diff = np.expand_dims(X_test, axis=1) - np.expand_dims(self.X_train, axis=0)

        # Apply the distance metric
        if distance == 'manhattan':
            # dist=(pred, train)
            dists = np.abs(diff).sum(axis=2)
        elif distance == 'euclidean':
            dists = np.sqrt((diff**2).sum(axis=2))
        else:
            raise NotImplementedError("Unknown distance metric '{}'.".format(distance))

        # i=(pred, K)
        # Find the K nearest neighours
        i = np.argpartition(dists, self.K, axis=1)[:, :self.K]

        # Return the class with the highest prob for each test sample
        return stats.mode(self.y_train[i], axis=1)[0]


def main(K: int, distance: str, test_portion: float) -> None:
    """
    Main method which loads the iris dataset, splits it into train/test sets
    and fits a KNN classifier to predict the correct class of examples.

    :K: number of nearest neighbors
    :distance: distance metric. Either 'euclidean' or 'manhattan'
    :test_portion: fraction of data we want to use for testing
    """
    # load the iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_portion,
                                                        random_state=28)

    # instantiate the KNN classifier
    knn = KNN(K)

    # fit the model to the training data
    knn.fit(X_train, y_train)

    # make the predictions on the test set
    preds = knn.predict(X_test, distance=distance)

    # compute the accuracy of your predictions
    accuracy = accuracy_score(y_test, preds)
    logging.info('Test accuracy (K=%d, d=%s): %.3f'%(K, distance, float(accuracy)))


if __name__=='__main__':
    cmdline_parser = argparse.ArgumentParser('KNN Classifier')

    # Required args
    cmdline_parser.add_argument('K',
                                type=int,
                                help="number of nearest neighbours")

    # Optional args
    cmdline_parser.add_argument('-d',
                                default='euclidean',
                                type=str,
                                choices=['euclidean', 'manhattan'],
                                help="distance metric")
    cmdline_parser.add_argument('-p',
                                default=0.3,
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

    main(K=args.K, distance=args.d, test_portion=args.p)
