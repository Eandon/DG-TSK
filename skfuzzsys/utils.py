"""
gate functions including the old ones and M-gate
threshold_fun
"""


def gate1(x):
    return 1 / (1 + (-x).exp())


def gate2(x):
    return 1 - (- x ** 2).exp()


def gate3(x):
    return (- x ** 2).exp()


def gate4(x):
    return x * (1 - x ** 2).exp().sqrt()


def gate_m(x):
    """
    M-gate function, it is shaped like an M
    :param x:
    :return:
    """
    return x ** 2 * (1 - x ** 2).exp()


def gate_m_derivative(x):
    return 2 * x * (1 - x ** 2).exp() - 2 * x.pow(3) * (1 - x ** 2).exp()


def threshold_fun(minimum, maximum, zeta):
    """
    computer the threshold for FS and RE
    threshold is between the minimum and maximum
    :param minimum:
    :param maximum:
    :param zeta: coefficient
    :return: threshold
    """
    return maximum - zeta * (maximum - minimum)
