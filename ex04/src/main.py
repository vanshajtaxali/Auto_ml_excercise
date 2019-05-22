import argparse
import logging
from functools import partial

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import utils
from matplotlib import pyplot as plt


def f(x):
    """
    Function to minimize. (Levy1D see https://www.sfu.ca/~ssurjano/levy.html). Global min value: 0.0
    """
    w0 = (1 + (x[0] - 1) / 4)
    term1 = np.power(np.sin(np.pi * w0), 2)

    term2 = 0
    for i in range(len(x) - 1):
        wi = 1 + (x[i] - 1) / 4
        term2 += np.power(wi - 1, 2) * (1 + 10 * np.power(np.sin(wi * np.pi + 1), 2))

    wd = (1 + (x[-1] - 1) / 4)
    term3 = np.power(wd - 1, 2)
    term3 *= (1 + np.power(np.sin(2 * np.pi * wd), 2))

    y = term1 + term2 + term3
    return y


def EI(x, model, eta, add=None):
    """
    Expected Improvement.
    :param x: point to determine the acquisition value
    :param model: GP to predict target function value
    :param eta: best so far seen value
    :param add: additional parameters necessary for the function
    """
    x = np.array([x]).reshape([-1, 1])
    m, s = model.predict(x, return_std=True)

    # Closed form solution from the slides
    # CDF is the cumulative density function
    # PDF is the probability density function
    diff = m - eta
    ei = diff * norm.cdf(diff / s) + s * norm.pdf(diff / s)
    return ei

def UCB(x, model, eta, add=None):
    """
    Upper Confidence Bound
    :param x: point to determine the acquisition value
    :param model: GP to predict target function value
    :param eta: best so far seen value
    :param add: additional parameters necessary for the function
    """
    x = np.array([x]).reshape([-1, 1])
    m, s = model.predict(x, return_std=True)
    k = add
    # stdev is subtracted due to our goal to minimize
    return m - k * s


def run_bo(acquisition, max_iter, init=25, random=True, acq_add=1, seed=1):
    """
    BO
    :param max_iter: max number of function calls
    :param init: number of points to build initial model
    :param seed: seed used to keep experiments reproducible
    :param random: if False initial points are linearly sampled in the bounds, otherwise uniformly random.
    :return: all evaluated points
    """
    # sample initial query points
    np.random.seed(seed)
    if random:
        x = np.random.uniform(-15, 10, init).reshape(-1, 1).tolist()
    else:
        x = np.linspace(-15, 10, init).reshape(-1, 1).tolist()
    # get corresponding response values
    y = list(map(f, x))
    incumbent = [y[0]]
    for val in y[1:]:
        if val < incumbent[-1]:
            incumbent.append(val)
        else:
            incumbent.append(incumbent[-1])

    for i in range(max_iter - init):  # BO loop
        logging.debug('Sample #%d' % (init + i))
        #Feel free to adjust the hyperparameters
        gp = Pipeline([["standardize", StandardScaler()],
                      ["GP", GPR(kernel=Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=10, random_state=seed)], 
                    ])
        gp.fit(x, y)  # fit the model

        # Partially initialize the acquisition function to work with the fmin interface
        # (only the x parameter is not specified)
        # TODO implement different acquisition functions
        acqui = partial(acquisition, model=gp, eta=min(y), add=acq_add)
        # optimize acquisition function, repeat 10 times, use best result
        x_ = None
        y_ = 10000
        # Feel free to adjust the hyperparameters
        for i in range(10):
            opt_res = minimize(acqui, np.random.uniform(-15, 10), bounds=[[-15, 10]], options={"maxfun": 10}, method="L-BFGS-B")
            if opt_res.fun[0] < y_:
                x_ = opt_res.x
                y_ = opt_res.fun[0]

        utils.plot_gp(gp, x, y, x_, acqui)

        x.append(x_)
        y.append(f(x_))
        if y[-1] < incumbent[-1]:
            incumbent.append(y[-1])
        else:
            incumbent.append(incumbent[-1])

    return y, incumbent

def main(num_evals, init_size, repetitions, random, seed):

    utils.enable_gp_plots = False
    # Plot target function
    # utils.plot_target()

    ei_inc = np.ndarray(shape=(repetitions, num_evals))
    ucb_inc = np.ndarray(shape=(repetitions, num_evals))
    rs_inc = np.ndarray(shape=(repetitions, num_evals))

    for i in range(repetitions):
        # utils.plot plots every step of the GP defined by the range object
        # range(2) means the first two steps will be plotted
        # range(0, 5, 2) means the first, third and fifth step will be plotted
        with utils.plot(range(2)):
            _, ei_inc[i, :] = run_bo(max_iter=num_evals, init=init_size, random=random, acquisition=EI, acq_add=1, seed=seed+i)
        with utils.plot(range(2)):
            _, ucb_inc[i, :] = run_bo(max_iter=num_evals, init=init_size, random=random, acquisition=UCB, acq_add=1, seed=seed+i)
        # Random search
        for j, x in enumerate(np.random.uniform(-15, 10, num_evals).reshape(-1, 1).tolist()):
            # for the incumbent, either append the function value when its better or the previous incumbent value
            if j == 0 or f(x) < rs_inc[i,j-1]:
                rs_inc[i, j] = f(x)
            else:
                rs_inc[i, j] = rs_inc[i, j-1]

    # TODO implement grid search
    # Gridsearch doesn't need repetition because it is not random
    gs_inc = []
    for x in np.linspace(-15, 10, num_evals).reshape(-1, 1).tolist():
        # for the incumbent, either append the function value when its better or the previous incumbent value
        if len(gs_inc) == 0 or f(x) < gs_inc[-1]:
            gs_inc.append(f(x))
        else:
            gs_inc.append(gs_inc[-1])


    # TODO evaluation
    x = np.arange(num_evals)
    ei_mean = ei_inc.mean(axis=0)
    ei_std = ei_inc.std(axis=0)
    ucb_mean = ucb_inc.mean(axis=0)
    ucb_std = ucb_inc.std(axis=0)
    rs_mean = rs_inc.mean(axis=0)
    rs_std = rs_inc.std(axis=0)

    plt.step(x, ei_mean, c='b', label="EI + std")
    plt.fill_between(x,
        ei_mean - ei_std,
        ei_mean + ei_std,
        alpha=.3, color='b', step='pre')

    plt.step(x, ucb_mean, c='g', label="UCB + std")
    plt.fill_between(x,
        ucb_mean - ucb_std,
        ucb_mean + ucb_std,
        alpha=.3, color='g', step='pre')
    plt.step(x, rs_mean, c='c', label="Randomsearch")
    plt.fill_between(x,
        rs_mean - rs_std,
        rs_mean + rs_std,
        alpha=.3, color='c', step='pre')
    plt.step(x, gs_inc, c='r', label="Gridsearch")
    plt.legend()
    plt.title("Incumbent")
    plt.xlabel("# function evaluations")
    plt.ylabel("best seen function value")
    plt.show()

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('ex06')

    cmdline_parser.add_argument('-n', '--num_func_evals',
                                default=100,
                                help='Number of function evaluations',
                                type=int)
    cmdline_parser.add_argument('-p', '--percentage_init',
                                default=0.25,
                                help='Percentage of budget (num_func_evals) to spend on building initial model',
                                type=float)
    cmdline_parser.add_argument('-r', '--random_initial_design',
                                action="store_true",
                                help='Use random initial points. If not set, initial points are sampled linearly on'
                                     ' the function bounds.')
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('--seed',
                                default=0,
                                help='Which seed to use',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('--repetitions',
                                default=5,
                                help='How often to repeat the experiment',
                                required=False,
                                type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    init_size = max(1, int(args.num_func_evals * args.percentage_init))
    main(num_evals=args.num_func_evals, init_size=init_size, repetitions=args.repetitions, random=args.random_initial_design, seed=args.seed)
