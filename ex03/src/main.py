import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt


def load_data(fl="data.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: y1, y2
    """
    data = np.loadtxt(fl, delimiter=",")
    y1 = data[:, 0]
    y2 = data[:, 1]
    return y1, y2


def paired_permutation_test(data_A, data_B, repetitions=10000) -> float:
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b
    :param repetitions: number of repetitions to use for the test
    :return:p-value
    """
    p_value = np.random.uniform(0, 1)
    return p_value


def cdf_plot(data_A, data_B):
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b  
    """
    
    # Preprocess data
    num_points = data_A.shape[0]
    y_axis = np.arange(num_points) / num_points

    sorted_data_A = np.sort(data_A)
    sorted_data_B = np.sort(data_B)

    # Plot the data
    plt.step(sorted_data_A, y_axis, 'b', label='A')
    plt.step(sorted_data_B, y_axis, 'y', label='B')
    plt.title('Performance of A and B')
    plt.xlabel('Error')
    plt.ylabel('P(error < X)')
    plt.legend()
    plt.savefig('step.png')
    plt.show()

    i1 = np.where(sorted_data_A < .4)[0][-1]
    i2 = np.where(sorted_data_B < .4)[0][-1]
    logging.info("A: P(error < {:.3}) = {:.3}".format(sorted_data_A[i1], y_axis[i1]))
    logging.info("B: P(error < {:.3}) = {:.3}".format(sorted_data_B[i2], y_axis[i2]))


def scatter_plot(data_A, data_B):
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b  
    """

    # Split the data
    diff = data_B - data_A
    i_cat1 = diff > .1
    i_cat2 = diff < -0.1
    i_cat3 = np.abs(diff) <= .1
    cat1_A = data_A[i_cat1]
    cat1_B = data_B[i_cat1]
    cat2_A = data_A[i_cat2]
    cat2_B = data_B[i_cat2]
    cat3_A = data_A[i_cat3]
    cat3_B = data_B[i_cat3]

    # Plot the data
    plt.scatter(cat1_A, cat1_B, marker='^', label='A is better (> .1)')
    plt.scatter(cat2_A, cat2_B, marker='v', label='B is better (> .1)')
    plt.scatter(cat3_A, cat3_B, marker='o', label='Otherwise')
    plt.title('Performance of A and B')
    plt.xlabel('Error value of A')
    plt.ylabel('Error value of B')
    plt.legend()
    plt.savefig('scatter.png')
    plt.show()

    # Log some information about the data
    logging.info('Name        | Number |')
    logging.info('------------+--------+-')
    logging.info('Overall     |{:8}|'.format(data_A.shape[0]))
    logging.info('A is better |{:8}|'.format(cat1_A.shape[0]))
    logging.info('B is better |{:8}|'.format(cat2_A.shape[0]))
    logging.info('Otherwise   |{:8}|'.format(cat3_A.shape[0]))
    logging.info('')
    logging.info('Dataset | mean | stdev | 25 percentile | median | 75 percentile')
    logging.info('--------+------+-------+---------------+--------+--------------')
    logging.info('A       |{:6.3}|{:7.3}|{:15.3}|{:8.3}|{:14.3}'.format(data_A.mean(), data_A.std(), np.quantile(data_A, .25), np.quantile(data_A, .5), np.quantile(data_A, .75)))
    logging.info('B       |{:6.3}|{:7.3}|{:15.3}|{:8.3}|{:14.3}'.format(data_B.mean(), data_B.std(), np.quantile(data_B, .25), np.quantile(data_B, .5), np.quantile(data_B, .75)))
    logging.info('')

def box_plot(data_A, data_B):
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b  
    """
    pass

def violin_plot(data_A, data_B):
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b  
    """
    pass


def main(args):
    logging.info('Loading data')
    data_A, data_B = load_data(fl="./data.csv")
    
    # (a)
    # TODO
    scatter_plot(data_A, data_B)

    # (b)
    # TODO
    cdf_plot(data_A, data_B)

    # (c)
    # TODO
    box_plot(data_A, data_B)
    violin_plot(data_A, data_B)

    # (d)
    # TODO
    alpha = None
    statistic = paired_permutation_test(data_A, data_B, repetitions=10000)

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('ex03')

    cmdline_parser.add_argument('-v', '--verbose', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity')
    cmdline_parser.add_argument('--seed', default=12345, help='Which seed to use', required=False, type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    np.random.seed(args.seed)
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    main(args)
