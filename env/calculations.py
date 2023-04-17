import numpy as np


def get_scalar(input_array):
    '''
        Calculates the scale of a number. For example, the numbers 500,30, 0.2 have scales of 100,10, 0.1 respectively
    '''

    def find_scalar(input):
        return 10 ** int("{:.2e}".format(input).split("e")[-1])

    return np.vectorize(find_scalar, otypes=[float])(input_array)

