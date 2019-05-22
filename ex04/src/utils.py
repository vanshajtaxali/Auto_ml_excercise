import contextlib

import numpy as np
from matplotlib import pyplot as plt

import main

enable_gp_plots = False
active_plots = []
max_plots = 0
current_plot = 0


def plot_gp(gp, obs_x, obs_y, next, acquisition):
    '''
    Initial source from: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
    '''
    if not enable_gp_plots:
        return

    xs = np.linspace(-15, 10, 100).reshape(-1, 1).tolist()
    y_pred, sigma = gp.predict(xs, return_std=True)

    global current_plot
    if current_plot not in active_plots.keys():
        current_plot += 1
        return
    plt.subplot(max_plots, 1, 1 + active_plots[current_plot])
    current_plot += 1

    plt.plot(xs, [main.f(x) for x in xs], 'r:', label=r'$f(x)$')
    plt.plot(obs_x, obs_y, 'r.', markersize=10, label=f'Observations ({len(obs_x)})')
    plt.plot(xs, y_pred, 'b-', label='GP Mean + Stddev')
    plt.fill(np.concatenate([xs, xs[::-1]]),
            np.concatenate([y_pred - sigma,
                            (y_pred + sigma)[::-1]]),
            alpha=.5, fc='b', ec='None')
    plt.axvline(x=next, c='r', ls='--', label='Next x')
    plt.plot(xs, [acquisition(x) for x in xs], 'g--', label="Acquisition")
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()

@contextlib.contextmanager
def plot(range_):
    global active_plots
    global current_plot
    global max_plots

    active_plots = { num: i for i, num in enumerate(range_) }
    current_plot = 0
    max_plots = len(active_plots)
    try:
        yield
    finally:
        plt.show()

def plot_target():
    xs = np.linspace(-15, 10, 100).reshape(-1, 1).tolist()
    ys = [main.f(x) for x in xs]
    plt.plot(xs, ys, 'r:')
    plt.title("Target function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig('target.png')
    plt.show()
