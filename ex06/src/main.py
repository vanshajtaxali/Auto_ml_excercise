import numpy as np
import matplotlib.pyplot as plt
import operator
import time

# ################################################### Data to use ######################################################
train_data = np.array([(np.array(x), np.sin(x) + 0.1 * np.cos(10 * x)) for x in np.linspace(0, 2 * np.pi, 64)])
X = np.array([np.array(x) for x in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
x, y = (zip(*train_data))
# ################################################### Data to use ######################################################


def phi_n_of_x(n: int) -> np.ndarray:
    """
    Basis function factory. Allows for easy generation of \phi_n(x).
    :param n: Dimensionality of the basis function
    :return: \phi_n(x)
    """

    def phi_of_x(x):
        return np.array([x ** i for i in range(n)])

    return phi_of_x


def gp_prediction_a(data: np.ndarray, X: np.ndarray, phi: callable, sigma_n: float, Sigma_p: np.ndarray) \
        -> (np.ndarray, np.ndarray):
    """
    Implementation of 2.11
    :param data: Data on which the model is fit
    :param X: Data to predict the mean and variance
    :param phi: basis functions
    :param sigma_n: variance for points in data
    :param Sigma_p: covariance
    :return: mean and variance for all points in X
    """
    # TODO compute parts
    # Phi: (N,n) matrix, N is the feature space after phi is applied, n is the number of observations
    Phi = np.apply_along_axis(phi, 0, data[:, 0])
    # A: (N,N) matrix
    A = sigma_n ** -2 * Phi @ Phi.T + np.linalg.inv(Sigma_p)
    A_inv = np.linalg.inv(A)
    # y: n-dim vector
    y = data[:, 1]

    tmp1 = A_inv @ Phi @ y

    # for storing the results
    means = []
    variances = []
    # loop over all X
    for xs in X:
        # TODO compute mean and variance
        phi_star =  phi(xs).reshape((-1,1))
        mean = sigma_n ** -2 * phi_star.T @ tmp1
        means.append(mean)
        variance = phi_star.T @ A_inv @ phi_star
        variances.append(variance.flatten())

    return np.array(means), np.array(variances)


def gp_prediction_b(data: np.ndarray, X: np.ndarray, phi: callable, sigma_n: float, Sigma_p: np.ndarray) \
        -> (np.ndarray, np.ndarray):
    """
    Implementation of 2.12
    :param data: Data on which the model is fit
    :param X: Data to predict the mean and variance
    :param phi: basis functions
    :param sigma_n: variance for points in data
    :param Sigma_p: covariance
    :return: mean and variance for all points in X
    """
    # TODO compute parts

    # for storing the results
    means = []
    variances = []
    # loop over all X
    for xs in X:
        pass
        # TODO compute mean and variance

    return np.array(means), np.array(variances)


# ##################################################### Exercise (b)
# Plot model predictions for the bayesian linear regression:
num_features = 2
means, variances = gp_prediction_a(train_data, x, phi_n_of_x(num_features), 1.0, np.eye(num_features))

# TODO plot model prediction
x_axis = np.arange(len(means))
plt.plot(x_axis, means, 'b-', label='GP Mean + Stddev')
plt.fill(np.concatenate([x_axis, x_axis[::-1]]),
    np.concatenate([means - variances, (means + variances)[::-1]]),
    alpha=.5, fc='b', ec='None')


plt.title('Bayesian linear regression example')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('plot1.png')


# ##################################################### Exercise (c)
# compare the predictions
mean1, var1 = gp_prediction_a(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))
mean2, var2 = gp_prediction_b(train_data, X, phi_n_of_x(num_features), 1.0, np.eye(num_features))
# TODO decide if mean and variance are equal
print('Equal means: ', '?')
print('Equal variances: ', '?')
print()

# plot the computation time
tsa = []
tsb = []
dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
for n in dimensions:
    print('Computing in feature dimension: ', n)
    I_n = np.eye(n)        # setup Identity of dimensionality n
    phi_n = phi_n_of_x(n)  # get basis functions

    # TODO measure time to compute for implementation A

    # TODO measure time to compute for implementation B

plt.figure()
plt.loglog(dimensions, tsa, label='Implementation 2.11', drawstyle='steps-post')
plt.loglog(dimensions, tsb, label='Implementation 2.12', drawstyle='steps-post')
plt.xlabel('input dimension')
plt.ylabel('runtime[sec]')
plt.legend(loc=2)

# plt.savefig('plot2.pdf')
plt.show()
